package com.example.picsprompt

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.providers.NNAPIFlags
import java.util.EnumSet
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.nio.FloatBuffer
import java.nio.LongBuffer
import java.util.regex.Pattern
import kotlin.math.floor
import kotlin.math.sqrt

class MainActivity : FlutterActivity() {

    companion object {
        private const val CHANNEL       = "picsprompt.inference"
        private const val BENCH_CHANNEL = "picsprompt.bench"
        private const val SEQ_LEN   = 77
        private const val SOT       = 49406L
        private const val EOT       = 49407L
        private const val VAE_SCALE = 0.18215f
        private const val TRAIN_T   = 1000
        /** Must match diffusers `LCMScheduler` default (`timestep_scaling`). */
        private const val LCM_TIMESTEP_SCALING = 10f
        private const val LCM_SIGMA_DATA = 0.5f
        /** Distillation grid length (`original_inference_steps` in LCMScheduler). */
        private const val LCM_ORIGINAL_INFERENCE_STEPS = 50
        /** LCM / DreamShaper-8-LCM expects low CFG (~1.5-2.5). Not SD 7.5. */
        private const val LCM_DEFAULT_GUIDANCE = 2.0f
        /** DDIM `steps_offset` from SD1.5 `scheduler_config.json` (matches diffusers PNDM/DDIM leading). */
        private const val DDIM_STEPS_OFFSET = 1
        /** Vanilla SD1.5 CFG range (LCM must stay low — see `LCM_DEFAULT_GUIDANCE`). */
        private const val SD_GUIDANCE_MIN = 1.0f
        /** High CFG + INT8 UNet often collapses to junk; FP32 bundles tolerate more. */
        private const val SD_GUIDANCE_MAX = 8.0f
    }

    /** Flutter `StandardMessageCodec` may send numbers as Int, Long, Double, or Float. */
    private fun anyToInt(v: Any?, default: Int): Int = when (v) {
        null -> default
        is Int -> v
        is Long -> v.coerceIn(Int.MIN_VALUE.toLong(), Int.MAX_VALUE.toLong()).toInt()
        is Double -> v.toInt()
        is Float -> v.toInt()
        is Number -> v.toInt()
        else -> default
    }

    private fun anyToFloat(v: Any?, default: Float): Float = when (v) {
        null -> default
        is Float -> v
        is Double -> v.toFloat()
        is Int -> v.toFloat()
        is Long -> v.toFloat()
        is Number -> v.toFloat()
        else -> default
    }

    private fun anyToLong(v: Any?, default: Long): Long = when (v) {
        null -> default
        is Long -> v
        is Int -> v.toLong()
        is Double -> v.toLong()
        is Float -> v.toLong()
        is Number -> v.toLong()
        else -> default
    }

    // OrtEnvironment is lightweight — keep it alive
    private var env: OrtEnvironment? = null

    // Model directory — set by initModel, used by generate
    private var modelDir: String? = null

    // Tokenizer — loaded once by initModel (tiny, ~2 MB). Must match HuggingFace `CLIPTokenizer`
    // (byte-level BPE + GPT-2-style merges). The old word-level BPE produced wrong `input_ids`
    // and the UNet effectively ignored the user prompt.
    private var vocab: Map<String, Int> = emptyMap()
    private var bpeRanks: Map<Pair<String, String>, Int> = emptyMap()
    private var byteEncoder: Map<Int, String> = emptyMap()
    private var unkTokenId: Int = 49407
    private lateinit var clipPat: Pattern
    private val clipBpeCache = mutableMapOf<String, String>()

    // ── Remote benchmark via adb intent extras ─────────────────────────────
    // Triggered by `picsprompt-models/scripts/run_remote_bench.sh`:
    //   adb shell am start -n com.example.picsprompt/.MainActivity \
    //     --es bench_model lcm --es bench_ep cpu --ei bench_iters 1
    // The Mac script tails logcat for `BENCHMARK_DONE:<path>` and pulls the
    // CSV. We finish the activity afterward so the next run starts from cold.
    private fun maybeStartRemoteBench(flutterEngine: FlutterEngine) {
        val extras = intent?.extras ?: return
        val model = extras.getString("bench_model") ?: return
        val ep = extras.getString("bench_ep") ?: "cpu"
        val iters = extras.getInt("bench_iters", 1)
        val warmup = extras.getInt("bench_warmup", 0)
        android.util.Log.i("Picsprompt", "remote-bench requested model=$model ep=$ep iters=$iters")

        val args = mapOf(
            "model" to model,
            "ep" to ep,
            "iters" to iters,
            "warmup" to warmup,
        )
        // Wait for Dart to register its handler; configureFlutterEngine fires
        // before main() finishes runApp, so a small post-delay is the cheapest
        // way to avoid MissingPluginException without plumbing readiness.
        android.os.Handler(mainLooper).postDelayed({
            MethodChannel(flutterEngine.dartExecutor.binaryMessenger, BENCH_CHANNEL)
                .invokeMethod("runBenchmark", args, object : MethodChannel.Result {
                    override fun success(result: Any?) {
                        android.util.Log.i("Picsprompt", "BENCHMARK_DONE:${result ?: ""}")
                        finishAndRemoveTask()
                    }
                    override fun error(code: String, message: String?, details: Any?) {
                        android.util.Log.e("Picsprompt", "BENCHMARK_FAILED:$code:$message")
                        finishAndRemoveTask()
                    }
                    override fun notImplemented() {
                        android.util.Log.e("Picsprompt", "BENCHMARK_FAILED:notImplemented")
                        finishAndRemoveTask()
                    }
                })
        }, 1500)
    }

    // ── Flutter engine wiring ──────────────────────────────────────────────

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        maybeStartRemoteBench(flutterEngine)
        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, CHANNEL)
            .setMethodCallHandler { call, result ->
                when (call.method) {
                    "initModel" -> {
                        val dir = call.argument<String>("modelDir")!!
                        Thread {
                            try {
                                initModel(dir)
                                result.success(null)
                            } catch (e: Exception) {
                                android.util.Log.e("Picsprompt", "initModel FAILED: ${e.javaClass.simpleName}: ${e.message}", e)
                                result.error("INIT_FAILED", "${e.javaClass.simpleName}: ${e.message}", null)
                            } catch (e: OutOfMemoryError) {
                                android.util.Log.e("Picsprompt", "initModel OOM", e)
                                result.error("INIT_FAILED", "OutOfMemoryError: not enough RAM", null)
                            }
                        }.start()
                    }
                    "generateImage" -> {
                        @Suppress("UNCHECKED_CAST")
                        val params = call.arguments as Map<String, Any>
                        currentEp = (params["executionProvider"] as? String)?.lowercase() ?: "cpu"
                        Thread {
                            try {
                                android.util.Log.i("Picsprompt", "generateImage start ep=$currentEp")
                                val path = generate(params)
                                android.util.Log.i("Picsprompt", "generateImage done: $path")
                                result.success(path)
                            } catch (e: Exception) {
                                android.util.Log.e("Picsprompt", "generateImage FAILED: ${e.javaClass.simpleName}: ${e.message}", e)
                                result.error("GEN_FAILED", "${e.javaClass.simpleName}: ${e.message}", null)
                            } catch (e: OutOfMemoryError) {
                                android.util.Log.e("Picsprompt", "generateImage OOM", e)
                                result.error("GEN_FAILED", "OutOfMemoryError", null)
                            }
                        }.start()
                    }
                    "generateAnimeGan" -> {
                        @Suppress("UNCHECKED_CAST")
                        val params = call.arguments as Map<String, Any>
                        currentEp = (params["executionProvider"] as? String)?.lowercase() ?: "cpu"
                        Thread {
                            try {
                                android.util.Log.i("Picsprompt", "generateAnimeGan start ep=$currentEp")
                                val path = generateAnimeGan(params)
                                android.util.Log.i("Picsprompt", "generateAnimeGan done: $path")
                                result.success(path)
                            } catch (e: Exception) {
                                android.util.Log.e("Picsprompt", "generateAnimeGan FAILED: ${e.javaClass.simpleName}: ${e.message}", e)
                                result.error("ANIMEGAN_FAILED", "${e.javaClass.simpleName}: ${e.message}", null)
                            } catch (e: OutOfMemoryError) {
                                android.util.Log.e("Picsprompt", "generateAnimeGan OOM", e)
                                result.error("ANIMEGAN_FAILED", "OutOfMemoryError", null)
                            }
                        }.start()
                    }
                    "generateLcmImg2Img" -> {
                        @Suppress("UNCHECKED_CAST")
                        val params = call.arguments as Map<String, Any>
                        currentEp = (params["executionProvider"] as? String)?.lowercase() ?: "cpu"
                        Thread {
                            try {
                                android.util.Log.i("Picsprompt", "generateLcmImg2Img start ep=$currentEp")
                                val path = generateLcmImg2Img(params)
                                android.util.Log.i("Picsprompt", "generateLcmImg2Img done: $path")
                                result.success(path)
                            } catch (e: Exception) {
                                android.util.Log.e("Picsprompt", "generateLcmImg2Img FAILED: ${e.javaClass.simpleName}: ${e.message}", e)
                                result.error("LCM_IMG2IMG_FAILED", "${e.javaClass.simpleName}: ${e.message}", null)
                            } catch (e: OutOfMemoryError) {
                                android.util.Log.e("Picsprompt", "generateLcmImg2Img OOM", e)
                                result.error("LCM_IMG2IMG_FAILED", "OutOfMemoryError", null)
                            }
                        }.start()
                    }
                    else -> result.notImplemented()
                }
            }
    }

    // ── Model initialisation ───────────────────────────────────────────────
    // Only loads tokenizer here — ONNX sessions are opened/closed per generation
    // to avoid holding all 990 MB in RAM simultaneously.

    private fun initModel(dir: String) {
        if (modelDir == dir) {
            android.util.Log.i("Picsprompt", "initModel — same dir, skipping")
            return
        }
        if (modelDir != null) {
            android.util.Log.i("Picsprompt", "initModel — switching bundle: $modelDir -> $dir")
            clipBpeCache.clear()
        }
        android.util.Log.i("Picsprompt", "initModel — dir: $dir")
        if (!File(dir).exists()) throw RuntimeException("Model directory not found: $dir")

        env = OrtEnvironment.getEnvironment()

        // Unconditional bundles (e.g. butterflies DDPM) ship only `unet/` — no
        // CLIP tokenizer to load. Detect by absence of `tokenizer/vocab.json`.
        if (File("$dir/tokenizer/vocab.json").exists()) {
            android.util.Log.i("Picsprompt", "Loading tokenizer...")
            loadTokenizer(dir)
        } else {
            android.util.Log.i("Picsprompt", "No tokenizer/ — treating as unconditional bundle")
        }
        modelDir = dir
        android.util.Log.i("Picsprompt", "initModel done — ONNX sessions will load per generation")
    }

    private fun loadTokenizer(dir: String) {
        val obj = JSONObject(File("$dir/tokenizer/vocab.json").readText())
        vocab = obj.keys().asSequence().associate { it to obj.getInt(it) }
        bpeRanks = File("$dir/tokenizer/merges.txt").readLines()
            .drop(1).filter { it.isNotBlank() }
            .mapIndexed { i, line ->
                val parts = line.trim().split(" ")
                Pair(parts[0], parts[1]) to i
            }.toMap()
        byteEncoder = bytesToUnicode()
        unkTokenId = vocab["<|endoftext|>"] ?: EOT.toInt()
        clipPat = Pattern.compile(
            """(?iu)<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
        )
        clipBpeCache.clear()
        android.util.Log.i("Picsprompt", "CLIP tokenizer: byteMap=${byteEncoder.size} merges=${bpeRanks.size} unk=$unkTokenId")
    }

    /** Same as `transformers.models.clip.tokenization_clip.bytes_to_unicode`. */
    private fun bytesToUnicode(): Map<Int, String> {
        val bs = mutableListOf<Int>()
        for (c in '!'.code..'~'.code) bs.add(c)
        for (c in '¡'.code..'¬'.code) bs.add(c)
        for (c in '®'.code..'ÿ'.code) bs.add(c)
        val cs = bs.toMutableList()
        var n = 0
        for (b in 0 until 256) {
            if (b !in bs) {
                bs.add(b)
                cs.add(256 + n)
                n++
            }
        }
        val out = HashMap<Int, String>(300)
        for (i in bs.indices) {
            val cp = cs[i]
            out[bs[i]] = if (Character.isBmpCodePoint(cp)) {
                cp.toChar().toString()
            } else {
                String(Character.toChars(cp))
            }
        }
        return out
    }

    private fun whitespaceClean(text: String): String =
        text.trim().replace(Regex("\\s+"), " ")

    /** HuggingFace `CLIPTokenizer.bpe` on a single byte-mapped token string. */
    private fun clipBpe(token: String): String {
        clipBpeCache[token]?.let { return it }
        if (token.isEmpty()) {
            val s = "</w>"
            clipBpeCache[token] = s
            return s
        }
        val word = ArrayList<String>()
        token.forEach { ch -> word.add(ch.toString()) }
        val lastSym = word.removeAt(word.lastIndex) + "</w>"
        word.add(lastSym)

        fun pairsOf(w: List<String>): MutableSet<Pair<String, String>> {
            val ps = mutableSetOf<Pair<String, String>>()
            for (i in 0 until w.size - 1) ps.add(w[i] to w[i + 1])
            return ps
        }

        var pairs = pairsOf(word)
        if (pairs.isEmpty()) {
            val s = token + "</w>"
            clipBpeCache[token] = s
            return s
        }
        while (true) {
            if (pairs.isEmpty()) break
            val bigram = pairs.minBy { bpeRanks.getOrDefault(it, Int.MAX_VALUE) }
            if (!bpeRanks.containsKey(bigram)) break
            val first = bigram.first
            val second = bigram.second
            val newWord = ArrayList<String>()
            var i = 0
            while (i < word.size) {
                var j = -1
                for (k in i until word.size) {
                    if (word[k] == first) {
                        j = k
                        break
                    }
                }
                if (j < 0) {
                    newWord.addAll(word.subList(i, word.size))
                    break
                }
                newWord.addAll(word.subList(i, j))
                i = j
                if (i < word.size - 1 && word[i] == first && word[i + 1] == second) {
                    newWord.add(first + second)
                    i += 2
                } else {
                    newWord.add(word[i])
                    i += 1
                }
            }
            word.clear()
            word.addAll(newWord)
            if (word.size <= 1) break
            pairs = pairsOf(word)
        }
        val joined = word.joinToString(" ")
        clipBpeCache[token] = joined
        return joined
    }

    /** HuggingFace `CLIPTokenizer._tokenize` (no ftfy path: whitespace clean + lower + regex + byte BPE). */
    private fun clipTokenIds(text: String): List<Int> {
        val t = whitespaceClean(text).lowercase()
        val out = ArrayList<Int>(32)
        val m = clipPat.matcher(t)
        while (m.find()) {
            val piece = m.group()
            val sb = StringBuilder(piece.length)
            for (b in piece.toByteArray()) {
                val sym = byteEncoder[b.toInt() and 0xFF]
                if (sym != null) sb.append(sym)
            }
            val bpePieces = clipBpe(sb.toString()).split(' ')
            for (p in bpePieces) {
                if (p.isEmpty()) continue
                out.add(vocab[p] ?: unkTokenId)
            }
        }
        return out
    }

    // ── Image generation ───────────────────────────────────────────────────
    // Loads each ONNX session only when needed, closes it immediately after.
    // Peak RAM = max(text_encoder=118MB, unet=823MB, vae=48MB) instead of all three.

    @Suppress("UNCHECKED_CAST")
    private fun generate(p: Map<String, Any>): String {
        val dir       = modelDir ?: throw IllegalStateException("initModel not called")
        val prompt    = p["prompt"]           as String
        val negPrompt = (p["negativePrompt"]  as? String) ?: ""
        val width     = anyToInt(p["width"], 256).coerceIn(64, 2048)
        val height    = anyToInt(p["height"], 256).coerceIn(64, 2048)
        val steps     = anyToInt(p["steps"], 8).coerceIn(1, 1000)
        val dirNorm = dir.replace('\\', '/').lowercase()
        val pipelineKind = (p["pipelineKind"] as? String)?.lowercase()

        // Unconditional DDPM (butterflies): no text encoder, no VAE, output is
        // pixel-space directly. Skip the entire SD pipeline.
        if (pipelineKind == "ddpm_unconditional") {
            return generateUnconditionalDdpm(
                dir = dir,
                width = width,
                height = height,
                steps = steps.coerceIn(1, 1000),
                seed = anyToLong(p["seed"], 42L),
                outPath = p["outputPath"] as String,
            )
        }
        val isStandardSd = when {
            pipelineKind == "standard_sd" -> true
            pipelineKind == "lcm" -> false
            // Older Flutter builds may omit pipelineKind; SD1.5 bundle path still contains "sd15".
            dirNorm.contains("/sd15/") || dirNorm.endsWith("/sd15") -> true
            else -> false
        }
        // LCM UNet needs low CFG; vanilla SD1.5 needs classical CFG (Dart sends ~5.5–8).
        val guidance = if (isStandardSd) {
            anyToFloat(p["guidance"], 7.5f).coerceIn(SD_GUIDANCE_MIN, SD_GUIDANCE_MAX)
        } else {
            anyToFloat(p["guidance"], LCM_DEFAULT_GUIDANCE).coerceIn(1.0f, 3.0f)
        }
        val seed      = anyToLong(p["seed"], 42L)
        val outPath   = p["outputPath"] as String

        val pShort = if (prompt.length > 120) prompt.take(120) + "..." else prompt
        android.util.Log.i(
            "Picsprompt",
            "generate params w=$width h=$height steps=$steps guidance=$guidance seed=$seed " +
                "pipeline=${if (isStandardSd) "standard_sd" else "lcm"}",
        )
        android.util.Log.i("Picsprompt", "prompt: $pShort")

        // SD1.5 DDIM uses more sequential UNet evals than LCM; modest extra threads help wall time.
        val opts = sessionOpts(intraOpThreads = if (isStandardSd) 4 else 2)
        val lH = height / 8
        val lW = width  / 8

        // ── Step 1: Text encoding — load encoder, encode, close ────────────
        android.util.Log.i("Picsprompt", "Step 1: encoding text...")
        val embeds: FloatArray
        val textEncSession = env!!.createSession("$dir/text_encoder/model.onnx", opts)
        try {
            embeds = encodeTextPair(textEncSession, negPrompt, prompt)
        } finally {
            textEncSession.close()
        }
        android.util.Log.i("Picsprompt", "Text encoder closed, RAM freed")

        // ── Step 2: Denoising — LCM (DreamShaper) vs DDIM (vanilla SD1.5 ONNX) ─
        val alphasCp = computeAlphasCumprod()
        val rng = java.util.Random(seed)
        var latents = FloatArray(4 * lH * lW) { rng.nextGaussian().toFloat() }

        val unetSession = env!!.createSession("$dir/unet/model.onnx", opts)
        try {
            if (isStandardSd) {
                val timesteps = ddimLeadingTimesteps(steps, DDIM_STEPS_OFFSET)
                android.util.Log.i("Picsprompt", "Step 2: denoising (${steps} DDIM / SD1.5 steps)...")
                for (s in timesteps.indices) {
                    val tCur = timesteps[s]
                    android.util.Log.i("Picsprompt", "  step ${s + 1}/${timesteps.size} t=$tCur")
                    val eps = runUnetEpsilonGuided(unetSession, latents, embeds, tCur, lH, lW, guidance)
                    latents = ddimStep(latents, eps, tCur, steps, alphasCp)
                }
            } else {
                val timesteps = lcmInferenceTimesteps(steps)
                android.util.Log.i("Picsprompt", "Step 2: denoising (${steps} LCM steps)...")
                for (s in timesteps.indices) {
                    val tCur = timesteps[s]
                    val tNext = if (s + 1 < timesteps.size) timesteps[s + 1] else -1
                    val isLast = (s == timesteps.lastIndex)
                    android.util.Log.i("Picsprompt", "  step ${s + 1}/${timesteps.size} t=$tCur -> $tNext")
                    latents = lcmSchedulerStep(
                        unetSession, latents, embeds, tCur, tNext, alphasCp,
                        guidance, lH, lW, isLast, rng,
                    )
                }
            }
        } finally {
            unetSession.close()
        }
        android.util.Log.i("Picsprompt", "UNet closed, RAM freed")

        // ── Step 3: VAE decode — load decoder, decode, close ──────────────
        android.util.Log.i("Picsprompt", "Step 3: VAE decoding...")
        val pixels: FloatArray
        val vaeSession = env!!.createSession("$dir/vae_decoder/model.onnx", opts)
        try {
            pixels = decodeVae(vaeSession, latents, height, width)
        } finally {
            vaeSession.close()
        }
        android.util.Log.i("Picsprompt", "VAE closed, saving image...")

        // ── Step 4: Save PNG ───────────────────────────────────────────────
        val bmp = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val px  = height * width
        for (row in 0 until height) for (col in 0 until width) {
            val r = pixels[0 * px + row * width + col].toInt().coerceIn(0, 255)
            val g = pixels[1 * px + row * width + col].toInt().coerceIn(0, 255)
            val b = pixels[2 * px + row * width + col].toInt().coerceIn(0, 255)
            bmp.setPixel(col, row, android.graphics.Color.rgb(r, g, b))
        }
        File(outPath).parentFile?.mkdirs()
        FileOutputStream(outPath).use { bmp.compress(Bitmap.CompressFormat.PNG, 100, it) }
        return outPath
    }

    private fun generateAnimeGan(p: Map<String, Any>): String {
        val modelPath = p["modelPath"] as String
        val imageBytes = p["imageBytes"] as ByteArray
        val outPath = p["outputPath"] as String
        if (!File(modelPath).exists()) throw RuntimeException("AnimeGAN model not found: $modelPath")

        val decoded = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
            ?: throw RuntimeException("Could not decode uploaded image")
        // Center-crop to square, then bilinear scale to 512. AnimeGANv2 was trained
        // on face-filling square crops; letterboxing leaves white borders the GAN
        // happily stylizes, which is what made earlier outputs look washed out.
        val squared = centerCropSquare(decoded, 512)
        val pixelsArgb = IntArray(512 * 512)
        squared.getPixels(pixelsArgb, 0, 512, 0, 0, 512, 512)

        val chw = FloatArray(3 * 512 * 512)
        val plane = 512 * 512
        for (i in 0 until plane) {
            val c = pixelsArgb[i]
            // (v/255 - 0.5)/0.5  ==  v/127.5 - 1.0
            chw[i] = ((c shr 16) and 0xFF) / 127.5f - 1f
            chw[plane + i] = ((c shr 8) and 0xFF) / 127.5f - 1f
            chw[2 * plane + i] = (c and 0xFF) / 127.5f - 1f
        }

        env = env ?: OrtEnvironment.getEnvironment()
        val session = env!!.createSession(modelPath, sessionOpts(intraOpThreads = 2))
        val tensor = OnnxTensor.createTensor(
            env!!,
            FloatBuffer.wrap(chw),
            longArrayOf(1, 3, 512, 512),
        )
        val raw: FloatArray
        try {
            val inputName = session.inputNames.first()
            val outputName = session.outputNames.first()
            val out = session.run(mapOf(inputName to tensor))
            try {
                val buf = (out.get(outputName).get() as OnnxTensor).floatBuffer
                buf.rewind()
                raw = FloatArray(buf.remaining()).also { buf.get(it) }
            } finally {
                out.close()
            }
        } finally {
            tensor.close()
            session.close()
        }

        val outArgb = IntArray(plane)
        for (i in 0 until plane) {
            val r = (((raw[i] + 1f) * 127.5f) + 0.5f).toInt().coerceIn(0, 255)
            val g = (((raw[plane + i] + 1f) * 127.5f) + 0.5f).toInt().coerceIn(0, 255)
            val b = (((raw[2 * plane + i] + 1f) * 127.5f) + 0.5f).toInt().coerceIn(0, 255)
            outArgb[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
        }
        val bmp = Bitmap.createBitmap(512, 512, Bitmap.Config.ARGB_8888)
        bmp.setPixels(outArgb, 0, 512, 0, 0, 512, 512)

        File(outPath).parentFile?.mkdirs()
        FileOutputStream(outPath).use { bmp.compress(Bitmap.CompressFormat.PNG, 100, it) }
        return outPath
    }

    /**
     * Img2img with DreamShaper-8-LCM (reuses the bundle already on device):
     *   center-crop the photo → VAE encode → noise to a partial timestep
     *   chosen from `strength` → run a few LCM denoising steps with the prompt
     *   conditioning the result → VAE decode → PNG.
     *
     * Quality and identity-preservation are controlled by `strength`:
     *   0.45–0.55 keeps face structure, 0.65–0.75 stylizes harder.
     */
    private fun generateLcmImg2Img(p: Map<String, Any>): String {
        val dir = modelDir ?: throw RuntimeException("initModel not called")
        val imageBytes = p["imageBytes"] as ByteArray
        val outPath = p["outputPath"] as String
        val prompt = (p["prompt"] as? String) ?: ""
        val negPrompt = (p["negativePrompt"] as? String) ?: ""
        val width = anyToInt(p["width"], 512).coerceIn(256, 768).let { (it / 8) * 8 }
        val height = anyToInt(p["height"], 512).coerceIn(256, 768).let { (it / 8) * 8 }
        val steps = anyToInt(p["steps"], 8).coerceIn(2, 50)
        val strength = anyToFloat(p["strength"], 0.55f).coerceIn(0.1f, 0.95f)
        val guidance = anyToFloat(p["guidance"], LCM_DEFAULT_GUIDANCE).coerceIn(1f, 3f)
        val seed = anyToLong(p["seed"], 42L)

        android.util.Log.i(
            "Picsprompt",
            "img2img params w=$width h=$height steps=$steps strength=$strength guidance=$guidance",
        )

        // ── Step 0: image → NCHW [-1, 1] ───────────────────────────────────
        val decoded = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
            ?: throw RuntimeException("Could not decode uploaded image")
        val sized = centerCropTo(decoded, width, height)
        val argb = IntArray(width * height)
        sized.getPixels(argb, 0, width, 0, 0, width, height)
        val plane = width * height
        val pixels = FloatArray(3 * plane)
        for (i in 0 until plane) {
            val c = argb[i]
            pixels[i] = ((c shr 16) and 0xFF) / 127.5f - 1f
            pixels[plane + i] = ((c shr 8) and 0xFF) / 127.5f - 1f
            pixels[2 * plane + i] = (c and 0xFF) / 127.5f - 1f
        }

        env = env ?: OrtEnvironment.getEnvironment()
        val opts = sessionOpts(intraOpThreads = 2)
        val lH = height / 8
        val lW = width / 8
        val latentSize = 4 * lH * lW

        // ── Step 1: text encode (CLIP) ─────────────────────────────────────
        android.util.Log.i("Picsprompt", "img2img Step 1: text encode")
        val embeds: FloatArray
        env!!.createSession("$dir/text_encoder/model.onnx", opts).use { s ->
            embeds = encodeTextPair(s, negPrompt, prompt)
        }

        // ── Step 2: VAE encode the photo into latents ──────────────────────
        android.util.Log.i("Picsprompt", "img2img Step 2: VAE encode")
        val initLatents: FloatArray
        env!!.createSession("$dir/vae_encoder/model.onnx", opts).use { s ->
            val inName = s.inputNames.first()
            val outName = s.outputNames.first()
            val tensor = OnnxTensor.createTensor(
                env!!,
                FloatBuffer.wrap(pixels),
                longArrayOf(1, 3, height.toLong(), width.toLong()),
            )
            try {
                val out = s.run(mapOf(inName to tensor))
                try {
                    val buf = (out.get(outName).get() as OnnxTensor).floatBuffer
                    buf.rewind()
                    val raw = FloatArray(buf.remaining()).also { buf.get(it) }
                    // Some ORT exports return a sampled latent (C=4); others return
                    // the Gaussian parameters mean‖logvar (C=8) in NCHW order, so
                    // the first half is the mean. Use the mean either way — picking
                    // the deterministic mode keeps the avatar stable across runs.
                    val mean = when (raw.size) {
                        latentSize -> raw
                        latentSize * 2 -> FloatArray(latentSize) { raw[it] }
                        else -> throw RuntimeException(
                            "VAE encoder gave ${raw.size} elems, expected " +
                                "$latentSize or ${latentSize * 2}",
                        )
                    }
                    initLatents = FloatArray(mean.size) { mean[it] * VAE_SCALE }
                } finally {
                    out.close()
                }
            } finally {
                tensor.close()
            }
        }

        // ── Step 3: noise the latents to the strength-chosen timestep ─────
        val alphasCp = computeAlphasCumprod()
        val rng = java.util.Random(seed)
        val fullTimesteps = lcmInferenceTimesteps(steps)
        val initStepCount = (steps * strength).toInt().coerceIn(1, steps)
        val tStart = (steps - initStepCount).coerceAtLeast(0)
        val activeTimesteps = fullTimesteps.copyOfRange(tStart, fullTimesteps.size)
        val firstT = activeTimesteps[0]
        val sqrtA = sqrt(alphasCp[firstT.coerceIn(0, TRAIN_T - 1)])
        val sqrtB = sqrt(1f - alphasCp[firstT.coerceIn(0, TRAIN_T - 1)])
        var latents = FloatArray(latentSize) { i ->
            sqrtA * initLatents[i] + sqrtB * rng.nextGaussian().toFloat()
        }
        android.util.Log.i(
            "Picsprompt",
            "img2img Step 3: ${activeTimesteps.size} LCM steps from t=$firstT",
        )

        // ── Step 4: denoise (LCM scheduler, exactly the same step kernel) ─
        env!!.createSession("$dir/unet/model.onnx", opts).use { s ->
            for (i in activeTimesteps.indices) {
                val tCur = activeTimesteps[i]
                val tNext = if (i + 1 < activeTimesteps.size) activeTimesteps[i + 1] else -1
                val isLast = (i == activeTimesteps.lastIndex)
                latents = lcmSchedulerStep(
                    s, latents, embeds, tCur, tNext, alphasCp,
                    guidance, lH, lW, isLast, rng,
                )
            }
        }

        // ── Step 5: VAE decode → PNG ───────────────────────────────────────
        android.util.Log.i("Picsprompt", "img2img Step 5: VAE decode")
        val outPixels: FloatArray
        env!!.createSession("$dir/vae_decoder/model.onnx", opts).use { s ->
            outPixels = decodeVae(s, latents, height, width)
        }

        val outArgb = IntArray(plane)
        for (row in 0 until height) for (col in 0 until width) {
            val idx = row * width + col
            val r = outPixels[idx].toInt().coerceIn(0, 255)
            val g = outPixels[plane + idx].toInt().coerceIn(0, 255)
            val b = outPixels[2 * plane + idx].toInt().coerceIn(0, 255)
            outArgb[idx] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
        }
        val bmp = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        bmp.setPixels(outArgb, 0, width, 0, 0, width, height)

        File(outPath).parentFile?.mkdirs()
        FileOutputStream(outPath).use { bmp.compress(Bitmap.CompressFormat.PNG, 100, it) }
        return outPath
    }

    private fun centerCropTo(src: Bitmap, targetW: Int, targetH: Int): Bitmap {
        val srcRatio = src.width.toFloat() / src.height
        val dstRatio = targetW.toFloat() / targetH
        val cropW: Int; val cropH: Int
        if (srcRatio > dstRatio) {
            cropH = src.height
            cropW = (cropH * dstRatio).toInt().coerceAtMost(src.width)
        } else {
            cropW = src.width
            cropH = (cropW / dstRatio).toInt().coerceAtMost(src.height)
        }
        val left = (src.width - cropW) / 2
        val top = (src.height - cropH) / 2
        var crop = Bitmap.createBitmap(src, left, top, cropW, cropH)
        // Multi-step downscale; same trick as centerCropSquare.
        while (crop.width > targetW * 2 && crop.height > targetH * 2) {
            val next = Bitmap.createScaledBitmap(
                crop, crop.width / 2, crop.height / 2, /*filter=*/ true,
            )
            if (next !== crop) crop.recycle()
            crop = next
        }
        return if (crop.width == targetW && crop.height == targetH) crop
        else Bitmap.createScaledBitmap(crop, targetW, targetH, /*filter=*/ true)
    }

    private fun centerCropSquare(src: Bitmap, size: Int): Bitmap {
        val side = minOf(src.width, src.height)
        val left = (src.width - side) / 2
        val top = (src.height - side) / 2
        var square = Bitmap.createBitmap(src, left, top, side, side)
        if (side == size) return square
        // Repeated 2× downscale before the final resize. Android's bilinear
        // filter aliases hard on big jumps (e.g. 3000 → 512); halving twice
        // first gives near-bicubic detail with the same primitives.
        while (square.width > size * 2) {
            val next = Bitmap.createScaledBitmap(
                square, square.width / 2, square.height / 2, /*filter=*/ true,
            )
            if (next !== square) square.recycle()
            square = next
        }
        return Bitmap.createScaledBitmap(square, size, size, /*filter=*/ true)
    }


    /**
     * Execution provider chosen for the *current* generation. Set per-call from
     * the Flutter method-channel arg (`executionProvider`). Defaults to CPU.
     *   cpu     — default ORT CPU EP (MLAS kernels)
     *   xnnpack — XNNPACK kernels via the XNNPACK EP (often 1.5–3× over CPU)
     *   nnapi   — Android NNAPI EP (routes to vendor NPU/DSP/GPU when supported,
     *             falls back to CPU per node otherwise)
     */
    @Volatile private var currentEp: String = "cpu"

    private fun sessionOpts(intraOpThreads: Int) = OrtSession.SessionOptions().apply {
        val threads = intraOpThreads.coerceIn(1, 6)
        setIntraOpNumThreads(threads)
        setOptimizationLevel(OrtSession.SessionOptions.OptLevel.BASIC_OPT)
        when (currentEp) {
            "xnnpack" -> {
                try {
                    addXnnpack(mapOf("intra_op_num_threads" to threads.toString()))
                    android.util.Log.i("Picsprompt", "EP: XNNPACK enabled (threads=$threads)")
                } catch (e: Throwable) {
                    android.util.Log.w("Picsprompt", "XNNPACK EP unavailable, falling back to CPU: ${e.message}")
                }
            }
            "nnapi" -> {
                try {
                    addNnapi(EnumSet.of(NNAPIFlags.USE_FP16))
                    android.util.Log.i("Picsprompt", "EP: NNAPI enabled (USE_FP16)")
                } catch (e: Throwable) {
                    android.util.Log.w("Picsprompt", "NNAPI EP unavailable, falling back to CPU: ${e.message}")
                }
            }
            else -> android.util.Log.i("Picsprompt", "EP: CPU (default)")
        }
    }

    // ── Text encoding ──────────────────────────────────────────────────────

    private fun encodeTextPair(session: OrtSession, uncond: String, cond: String): FloatArray {
        val eA = encodeOneText(session, uncond)
        val eB = encodeOneText(session, cond)
        return FloatArray(eA.size + eB.size) { i -> if (i < eA.size) eA[i] else eB[i - eA.size] }
    }

    private fun encodeOneText(session: OrtSession, text: String): FloatArray {
        val ids    = buildInputIds(text)
        val tensor = OnnxTensor.createTensor(env!!, LongBuffer.wrap(ids), longArrayOf(1, SEQ_LEN.toLong()))
        val out    = session.run(mapOf("input_ids" to tensor))
        val flat   = (out.get("last_hidden_state").get() as OnnxTensor).floatBuffer
        flat.rewind()
        val arr    = FloatArray(flat.remaining()).also { flat.get(it) }
        out.close(); tensor.close()
        return arr
    }

    private fun buildInputIds(text: String): LongArray {
        val tokens = clipTokenIds(text)
        val ids    = LongArray(SEQ_LEN) { EOT }
        ids[0] = SOT
        val end = minOf(tokens.size, SEQ_LEN - 2)
        for (i in 0 until end) ids[i + 1] = tokens[i].toLong()
        ids[end + 1] = EOT
        return ids
    }

    // ── UNet forward + CFG (shared by LCM and DDIM paths) ───────────────────

    private fun runUnetEpsilonGuided(
        session: OrtSession,
        sample: FloatArray,
        embeds: FloatArray,
        timestep: Int,
        lH: Int,
        lW: Int,
        guidance: Float,
    ): FloatArray {
        val latentSize = 4 * lH * lW
        val xBatch = FloatArray(2 * latentSize) { i -> sample[i % latentSize] }
        val sT = OnnxTensor.createTensor(env!!, FloatBuffer.wrap(xBatch), longArrayOf(2, 4, lH.toLong(), lW.toLong()))
        val tT = OnnxTensor.createTensor(env!!, FloatBuffer.wrap(floatArrayOf(timestep.toFloat())), longArrayOf())
        val eT = OnnxTensor.createTensor(env!!, FloatBuffer.wrap(embeds), longArrayOf(2, SEQ_LEN.toLong(), 768))

        val out = session.run(mapOf("sample" to sT, "timestep" to tT, "encoder_hidden_states" to eT))
        val predBuf = (out.get("out_sample").get() as OnnxTensor).floatBuffer
        predBuf.rewind()
        val pred = FloatArray(predBuf.remaining()).also { predBuf.get(it) }
        out.close(); sT.close(); tT.close(); eT.close()

        return FloatArray(latentSize) { i ->
            pred[i] + guidance * (pred[latentSize + i] - pred[i])
        }
    }

    // ── DDIM (vanilla SD1.5 UNet — matches diffusers DDIM leading + offset) ─

    /**
     * Same as `DDIMScheduler.set_timesteps` with `timestep_spacing="leading"` and `steps_offset`.
     */
    private fun ddimLeadingTimesteps(numInferenceSteps: Int, stepsOffset: Int): IntArray {
        val stepRatio = TRAIN_T / numInferenceSteps
        return IntArray(numInferenceSteps) { s ->
            val tRaw = (numInferenceSteps - 1 - s) * stepRatio
            (tRaw + stepsOffset).coerceIn(0, TRAIN_T - 1)
        }
    }

    /**
     * One DDIM step with `eta=0` (deterministic), `prediction_type=epsilon`, `clip_sample=false`,
     * `set_alpha_to_one=false` → `alpha_prod_t_prev` at the end uses `alphas_cumprod[0]`.
     */
    private fun ddimStep(
        sample: FloatArray,
        modelOutput: FloatArray,
        timestep: Int,
        numInferenceSteps: Int,
        alphasCp: FloatArray,
    ): FloatArray {
        val stepRatio = TRAIN_T / numInferenceSteps
        val prevTimestep = timestep - stepRatio

        val alphaProdT = alphasCp[timestep.coerceIn(0, TRAIN_T - 1)]
        val alphaProdTPrev = if (prevTimestep >= 0) {
            alphasCp[prevTimestep.coerceIn(0, TRAIN_T - 1)]
        } else {
            alphasCp[0]
        }
        val betaProdT = 1f - alphaProdT

        val sqrtAt = sqrt(alphaProdT)
        val sqrtBt = sqrt(betaProdT)
        val predOriginal = FloatArray(sample.size) { i ->
            (sample[i] - sqrtBt * modelOutput[i]) / sqrtAt
        }

        val sqrtAprev = sqrt(alphaProdTPrev)
        val inside = (1f - alphaProdTPrev).coerceAtLeast(0f)
        val coeffDir = sqrt(inside)
        return FloatArray(sample.size) { i ->
            sqrtAprev * predOriginal[i] + coeffDir * modelOutput[i]
        }
    }

    // ── LCM scheduler (matches diffusers `scheduling_lcm.LCMScheduler`) ───

    /**
     * Same construction as `LCMScheduler.set_timesteps` for the non-custom case:
     * lcm_origin = [(1..original_steps).map { it * k - 1 }] reversed, then
     * `floor(linspace(0, n, num_steps, endpoint=false))` as indices.
     */
    private fun lcmInferenceTimesteps(numSteps: Int): IntArray {
        val originalSteps = LCM_ORIGINAL_INFERENCE_STEPS
        val k = TRAIN_T / originalSteps
        val originAsc = IntArray(originalSteps) { i -> (i + 1) * k - 1 }
        val reversed = originAsc.reversedArray()
        val n = reversed.size
        return IntArray(numSteps) { s ->
            val idx = floor((s.toDouble() * n) / numSteps).toInt().coerceAtMost(n - 1)
            reversed[idx]
        }
    }

    private fun computeAlphasCumprod(): FloatArray {
        val s = sqrt(0.00085f); val e = sqrt(0.012f); var cp = 1f
        return FloatArray(TRAIN_T) { i ->
            val t = i.toFloat() / (TRAIN_T - 1)
            val b = (s + t * (e - s)).let { it * it }
            cp *= (1f - b); cp
        }
    }

    /**
     * One LCM `step` (epsilon prediction + boundary scalings + optional noise),
     * matching `LCMScheduler.step` in diffusers 0.37+.
     */
    private fun lcmSchedulerStep(
        session: OrtSession,
        sample: FloatArray,
        embeds: FloatArray,
        timestep: Int,
        prevTimestep: Int,
        alphasCp: FloatArray,
        guidance: Float,
        lH: Int,
        lW: Int,
        isLastSchedulerStep: Boolean,
        rng: java.util.Random,
    ): FloatArray {
        val latentSize = 4 * lH * lW
        val modelOutput = runUnetEpsilonGuided(session, sample, embeds, timestep, lH, lW, guidance)

        val alphaProdT = alphasCp[timestep.coerceIn(0, TRAIN_T - 1)]
        val alphaProdTPrev = if (prevTimestep < 0) {
            1f
        } else {
            alphasCp[prevTimestep.coerceIn(0, TRAIN_T - 1)]
        }
        val betaProdT = 1f - alphaProdT
        val betaProdTPrev = 1f - alphaProdTPrev

        val scaledT = timestep.toFloat() * LCM_TIMESTEP_SCALING
        val sd = LCM_SIGMA_DATA
        val cSkip = (sd * sd) / (scaledT * scaledT + sd * sd)
        val cOut = scaledT / sqrt(scaledT * scaledT + sd * sd)

        val sqrtAt = sqrt(alphaProdT)
        val sqrtBt = sqrt(betaProdT)
        val predOrig = FloatArray(latentSize) { i ->
            (sample[i] - sqrtBt * modelOutput[i]) / sqrtAt
        }

        val denoised = FloatArray(latentSize) { i ->
            cOut * predOrig[i] + cSkip * sample[i]
        }

        if (isLastSchedulerStep) {
            return denoised
        }

        val sqrtAtp = sqrt(alphaProdTPrev)
        val sqrtBtp = sqrt(betaProdTPrev)
        return FloatArray(latentSize) { i ->
            sqrtAtp * denoised[i] + sqrtBtp * rng.nextGaussian().toFloat()
        }
    }

    // ── VAE decode ─────────────────────────────────────────────────────────

    // ── Unconditional DDPM (butterflies-128) ───────────────────────────────
    //
    // The model is a UNet2DModel that maps `(noisy_image[1,3,H,W], timestep[1])`
    // to predicted noise of the same shape. There is no text encoder, no VAE,
    // and no CFG — output of the UNet is pixel-space directly (range [-1, 1]
    // when fully denoised). We run a standard DDPM step from `T-1` down to `0`
    // with `leading` timestep spacing to match diffusers training.
    private fun generateUnconditionalDdpm(
        dir: String,
        width: Int,
        height: Int,
        steps: Int,
        seed: Long,
        outPath: String,
    ): String {
        android.util.Log.i(
            "Picsprompt",
            "ddpm w=$width h=$height steps=$steps seed=$seed",
        )
        val opts = sessionOpts(intraOpThreads = 4)
        val alphasCp = computeLinearAlphasCumprod()
        val timesteps = ddpmLeadingTimesteps(steps)
        val rng = java.util.Random(seed)

        // Initial latent = pure Gaussian noise at full image resolution.
        var sample = FloatArray(3 * height * width) { rng.nextGaussian().toFloat() }

        val unetSession = env!!.createSession("$dir/unet/model.onnx", opts)
        try {
            for (s in timesteps.indices) {
                val tCur = timesteps[s]
                val tPrev = if (s + 1 < timesteps.size) timesteps[s + 1] else -1
                if (s % 5 == 0 || s == timesteps.lastIndex) {
                    android.util.Log.i(
                        "Picsprompt",
                        "  ddpm step ${s + 1}/${timesteps.size} t=$tCur -> $tPrev",
                    )
                }
                val eps = runUnconditionalUnet(unetSession, sample, tCur, height, width)
                sample = ddpmStep(sample, eps, tCur, tPrev, alphasCp, rng)
            }
        } finally {
            unetSession.close()
        }
        android.util.Log.i("Picsprompt", "ddpm UNet closed, saving image...")

        // sample is in roughly [-1, 1]. Rescale to [0, 255] and clamp.
        val bmp = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val px = height * width
        for (row in 0 until height) for (col in 0 until width) {
            val r = (((sample[0 * px + row * width + col] + 1f) / 2f).coerceIn(0f, 1f) * 255f).toInt()
            val g = (((sample[1 * px + row * width + col] + 1f) / 2f).coerceIn(0f, 1f) * 255f).toInt()
            val b = (((sample[2 * px + row * width + col] + 1f) / 2f).coerceIn(0f, 1f) * 255f).toInt()
            bmp.setPixel(col, row, android.graphics.Color.rgb(r, g, b))
        }
        File(outPath).parentFile?.mkdirs()
        FileOutputStream(outPath).use { bmp.compress(Bitmap.CompressFormat.PNG, 100, it) }
        return outPath
    }

    /**
     * One UNet eval. Input names match `butterflies/export_onnx.py`:
     *   sample   : float32 [1, 3, H, W]
     *   timestep : int64   [1]
     * Output:
     *   noise_pred : float32 [1, 3, H, W]
     */
    private fun runUnconditionalUnet(
        session: OrtSession,
        sample: FloatArray,
        timestep: Int,
        height: Int,
        width: Int,
    ): FloatArray {
        val sampleTensor = OnnxTensor.createTensor(
            env!!,
            FloatBuffer.wrap(sample),
            longArrayOf(1, 3, height.toLong(), width.toLong()),
        )
        val tsTensor = OnnxTensor.createTensor(
            env!!,
            LongBuffer.wrap(longArrayOf(timestep.toLong())),
            longArrayOf(1),
        )
        try {
            val out = session.run(mapOf("sample" to sampleTensor, "timestep" to tsTensor))
            try {
                val outName = session.outputNames.first()
                val buf = (out.get(outName).get() as OnnxTensor).floatBuffer
                buf.rewind()
                return FloatArray(buf.remaining()).also { buf.get(it) }
            } finally {
                out.close()
            }
        } finally {
            sampleTensor.close()
            tsTensor.close()
        }
    }

    /**
     * Linear beta schedule (`beta_start=1e-4`, `beta_end=0.02`) — matches the
     * butterflies tutorial's `DDPMScheduler` defaults. Different from the SD
     * `scaled_linear` schedule used by `computeAlphasCumprod()`.
     */
    private fun computeLinearAlphasCumprod(): FloatArray {
        val bs = 1e-4f
        val be = 0.02f
        var cp = 1f
        return FloatArray(TRAIN_T) { i ->
            val b = bs + (be - bs) * i.toFloat() / (TRAIN_T - 1)
            cp *= (1f - b); cp
        }
    }

    /**
     * `DDPMScheduler.set_timesteps` with `timestep_spacing="leading"`:
     *   step = T // N
     *   timesteps = (arange(0, N) * step).reverse()
     * For N=50, T=1000 → [980, 960, …, 20, 0].
     */
    private fun ddpmLeadingTimesteps(numSteps: Int): IntArray {
        val n = numSteps.coerceIn(1, TRAIN_T)
        val step = TRAIN_T / n
        return IntArray(n) { i -> (n - 1 - i) * step }
    }

    /**
     * One DDPM denoising step (epsilon prediction, `clip_sample=True`,
     * `variance_type="fixed_small"`). Matches `DDPMScheduler.step` in
     * diffusers 0.37+.
     */
    private fun ddpmStep(
        sample: FloatArray,
        eps: FloatArray,
        t: Int,
        prevT: Int,
        alphasCp: FloatArray,
        rng: java.util.Random,
    ): FloatArray {
        val alphaProdT = alphasCp[t.coerceIn(0, TRAIN_T - 1)]
        val alphaProdTPrev = if (prevT < 0) 1f else alphasCp[prevT.coerceIn(0, TRAIN_T - 1)]
        val betaProdT = 1f - alphaProdT
        val betaProdTPrev = 1f - alphaProdTPrev
        val currentAlphaT = alphaProdT / alphaProdTPrev
        val currentBetaT = 1f - currentAlphaT

        val sqrtAlphaProdT = sqrt(alphaProdT)
        val sqrtBetaProdT = sqrt(betaProdT)
        val coefPredOrig = sqrt(alphaProdTPrev) * currentBetaT / betaProdT
        val coefSample = sqrt(currentAlphaT) * betaProdTPrev / betaProdT

        // Variance for fixed_small (clamped per diffusers).
        val rawVar = (betaProdTPrev / betaProdT) * currentBetaT
        val variance = rawVar.coerceAtLeast(1e-20f)
        val sigma = sqrt(variance)
        val isLast = prevT < 0

        val out = FloatArray(sample.size)
        for (i in sample.indices) {
            var pred = (sample[i] - sqrtBetaProdT * eps[i]) / sqrtAlphaProdT
            if (pred < -1f) pred = -1f else if (pred > 1f) pred = 1f
            val mean = coefPredOrig * pred + coefSample * sample[i]
            out[i] = if (isLast) mean else mean + sigma * rng.nextGaussian().toFloat()
        }
        return out
    }

    private fun decodeVae(session: OrtSession, latents: FloatArray, height: Int, width: Int): FloatArray {
        val lH = height / 8; val lW = width / 8
        val scaled = FloatArray(latents.size) { latents[it] / VAE_SCALE }
        val tensor = OnnxTensor.createTensor(env!!, FloatBuffer.wrap(scaled), longArrayOf(1, 4, lH.toLong(), lW.toLong()))
        val out    = session.run(mapOf("latent_sample" to tensor))
        val rawBuf = (out.get("sample").get() as OnnxTensor).floatBuffer
        rawBuf.rewind()
        val raw    = FloatArray(rawBuf.remaining()).also { rawBuf.get(it) }
        out.close(); tensor.close()
        return FloatArray(raw.size) { i -> ((raw[i] + 1f) / 2f).coerceIn(0f, 1f) * 255f }
    }
}
