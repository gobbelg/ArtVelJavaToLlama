package core;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Settings used to start llama-server. Instances are normally built from the
 * settings file by AppSettings.serverConfig(); the values for each computer
 * (paths, --ctx-size, --threads, GPU layers, extra flags) live in that file, not
 * in the code. With the Xeon + A2 values it reproduces this command line:
 *
 *   llama-server.exe --host 127.0.0.1 --port 8080 --model "<gguf>"
 *       --n-gpu-layers 999 --ctx-size 49152 --parallel 4 --cont-batching
 *       --batch-size 2048 --ubatch-size 512 --flash-attn on --threads 12
 *       --load-mode mlock
 *
 * SLOT SIZE. llama-server divides --ctx-size equally among the --parallel slots,
 * and each request (system prompt + report + generated output) must fit in ONE
 * slot. So the context available to a single document is
 *
 *       slotSize() = ctxSize / parallel        (49152 / 4 = 12288 tokens)
 *
 * This is calculated from the start-up parameters; the running server is not
 * asked. (LlamaServerManager additionally cross-checks the n_ctx_slot value the
 * server prints in its own start-up log.)
 *
 * Instances are immutable. withParallel() keeps --ctx-size fixed, so fewer
 * slots means LARGER slots at the same KV-cache cost; withSlots() keeps the
 * per-slot size fixed instead (used by the throughput harness).
 *
 * Optional flags: threads <= 0 omits --threads (the server chooses);
 * contBatching false omits --cont-batching. Anything else (for example
 * --cache-type-k q8_0, or --load-mode mlock) goes in extraArgs.
 */
public final class LlamaServerConfig {

    private final Path executable;
    private final String host;
    private final int port;
    private final Path model;
    private final int nGpuLayers;
    private final int ctxSize;
    private final int parallel;
    private final boolean contBatching;
    private final int batchSize;
    private final int ubatchSize;
    private final int threads;
    private final String flashAttn;
    private final List<String> extraArgs;
    private final int startupTimeoutMinutes;

    public LlamaServerConfig(Path executable, String host, int port, Path model,
            int nGpuLayers, int ctxSize, int parallel, boolean contBatching,
            int batchSize, int ubatchSize, int threads, String flashAttn,
            List<String> extraArgs, int startupTimeoutMinutes) {
        if (parallel < 1) {
            throw new IllegalArgumentException("parallel must be at least 1: " + parallel);
        }
        if (ctxSize < parallel) {
            throw new IllegalArgumentException(
                    "ctxSize (" + ctxSize + ") must be at least parallel (" + parallel + ")");
        }
        this.executable = executable;
        this.host = host;
        this.port = port;
        this.model = model;
        this.nGpuLayers = nGpuLayers;
        this.ctxSize = ctxSize;
        this.parallel = parallel;
        this.contBatching = contBatching;
        this.batchSize = batchSize;
        this.ubatchSize = ubatchSize;
        this.threads = threads;
        this.flashAttn = flashAttn;
        this.extraArgs = List.copyOf(extraArgs);
        this.startupTimeoutMinutes = startupTimeoutMinutes;
    }

    // -------------------------------------------------------------------------

    public Path executable()         { return executable; }
    public String host()             { return host; }
    public int port()                { return port; }
    public Path model()              { return model; }
    public int ctxSize()             { return ctxSize; }
    public int parallel()            { return parallel; }

    /** How long LlamaServerManager waits for the model to load and /health to report ready. */
    public int startupTimeoutMinutes() { return startupTimeoutMinutes; }

    /** Tokens available to ONE request: --ctx-size divided by --parallel. */
    public int slotSize() {
        return ctxSize / parallel;
    }

    public String baseUrl() {
        return "http://" + host + ":" + port;
    }

    /** Same settings with a different --parallel; --ctx-size is unchanged, so slots get larger as parallel drops. */
    public LlamaServerConfig withParallel(int newParallel) {
        return new LlamaServerConfig(executable, host, port, model, nGpuLayers, ctxSize,
                newParallel, contBatching, batchSize, ubatchSize, threads, flashAttn,
                extraArgs, startupTimeoutMinutes);
    }

    /** Same settings with N slots of exactly slotSizeTokens each (--ctx-size = N x slot size). */
    public LlamaServerConfig withSlots(int newParallel, int slotSizeTokens) {
        return new LlamaServerConfig(executable, host, port, model, nGpuLayers,
                newParallel * slotSizeTokens, newParallel, contBatching, batchSize,
                ubatchSize, threads, flashAttn, extraArgs, startupTimeoutMinutes);
    }

    /** The full command: executable followed by its arguments, in the original order. */
    public List<String> commandLine() {
        List<String> c = new ArrayList<>();
        c.add(executable.toString());
        c.add("--host");         c.add(host);
        c.add("--port");         c.add(Integer.toString(port));
        c.add("--model");        c.add(model.toString());
        c.add("--n-gpu-layers"); c.add(Integer.toString(nGpuLayers));
        c.add("--ctx-size");     c.add(Integer.toString(ctxSize));
        c.add("--parallel");     c.add(Integer.toString(parallel));
        if (contBatching) {
            c.add("--cont-batching");
        }
        c.add("--batch-size");   c.add(Integer.toString(batchSize));
        c.add("--ubatch-size");  c.add(Integer.toString(ubatchSize));
        c.add("--flash-attn");   c.add(flashAttn);
        if (threads > 0) {
            c.add("--threads");  c.add(Integer.toString(threads));
        }
        c.addAll(extraArgs);
        return c;
    }

    public String describe() {
        return "--parallel " + parallel + " --ctx-size " + ctxSize
                + " (slot size " + slotSize() + " tokens)";
    }
}
