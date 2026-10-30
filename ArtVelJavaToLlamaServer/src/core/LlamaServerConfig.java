package core;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

/**
 * Settings used to start llama-server. vaDefaults() reproduces this command line:
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
 * slots means LARGER slots at the same KV-cache (VRAM) cost; withSlots() keeps
 * the per-slot size fixed instead (used by the throughput harness).
 */
public final class LlamaServerConfig {

    private final Path executable;
    private final String host;
    private final int port;
    private final Path model;
    private final int nGpuLayers;
    private final int ctxSize;
    private final int parallel;
    private final int batchSize;
    private final int ubatchSize;
    private final int threads;
    private final String flashAttn;
    private final List<String> extraArgs;

    public LlamaServerConfig(Path executable, String host, int port, Path model,
            int nGpuLayers, int ctxSize, int parallel, int batchSize, int ubatchSize,
            int threads, String flashAttn, List<String> extraArgs) {
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
        this.batchSize = batchSize;
        this.ubatchSize = ubatchSize;
        this.threads = threads;
        this.flashAttn = flashAttn;
        this.extraArgs = List.copyOf(extraArgs);
    }

    /** The configuration from the VA command line (edit the paths/values here). */
    public static LlamaServerConfig vaDefaults() {
        return new LlamaServerConfig(
                Paths.get("P:\\ORD_Nguyen_202206062D\\Glenn\\LlamaAppInstall_261008"
                        + "\\llama-b11514-bin-win-cuda-12.4-x64\\llama-server.exe"),
                "127.0.0.1",
                8080,
                Paths.get("O:\\OMHSP_CONDUIT\\NLP\\Glenn\\LLMs\\Models"
                        + "\\gemma-4-E4B-it-IQ4_NL\\gemma-4-E4B-it-IQ4_NL.gguf"),
                999,      // --n-gpu-layers
                49152,    // --ctx-size (TOTAL across all slots)
                4,        // --parallel
                2048,     // --batch-size
                512,      // --ubatch-size
                12,       // --threads
                "on",     // --flash-attn
                // Passed through exactly as in the original command. If your
                // llama-server build does not accept it, the server exits at
                // start-up and the error is shown; "--mlock" is the older spelling.
                List.of("--load-mode", "mlock"));
    }

    // -------------------------------------------------------------------------

    public Path executable()         { return executable; }
    public String host()             { return host; }
    public int port()                { return port; }
    public Path model()              { return model; }
    public int ctxSize()             { return ctxSize; }
    public int parallel()            { return parallel; }

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
                newParallel, batchSize, ubatchSize, threads, flashAttn, extraArgs);
    }

    /** Same settings with N slots of exactly slotSizeTokens each (--ctx-size = N x slot size). */
    public LlamaServerConfig withSlots(int newParallel, int slotSizeTokens) {
        return new LlamaServerConfig(executable, host, port, model, nGpuLayers,
                newParallel * slotSizeTokens, newParallel, batchSize, ubatchSize, threads,
                flashAttn, extraArgs);
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
        c.add("--cont-batching");
        c.add("--batch-size");   c.add(Integer.toString(batchSize));
        c.add("--ubatch-size");  c.add(Integer.toString(ubatchSize));
        c.add("--flash-attn");   c.add(flashAttn);
        c.add("--threads");      c.add(Integer.toString(threads));
        c.addAll(extraArgs);
        return c;
    }

    public String describe() {
        return "--parallel " + parallel + " --ctx-size " + ctxSize
                + " (slot size " + slotSize() + " tokens)";
    }
}
