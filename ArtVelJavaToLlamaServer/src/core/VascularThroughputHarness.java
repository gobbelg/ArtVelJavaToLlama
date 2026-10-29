package core;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;
import java.util.concurrent.atomic.AtomicInteger;

import core.VascularModels.VascularResult;

/**
 * Standalone throughput harness.
 *
 * Sweeps the client-side THREADS value across a range and, for each setting,
 * runs a fixed sample of documents through the extractor and reports aggregate
 * documents-per-minute. Use this to find the point at which adding concurrency
 * stops improving throughput on the NVIDIA A2.
 *
 * IMPORTANT — this tunes BOTH sides of the pipe, which must be swept together:
 *   1. Client THREADS  = the sweep value below (concurrent in-flight requests).
 *   2. Server --parallel = MUST be set to the SAME value when you start
 *      llama-server, because client threads beyond --parallel just queue on the
 *      server (no gain) and fewer starve the server's KV-cache slots.
 *   3. Server --ctx-size = N x 12288 (so every slot keeps 12288 tokens regardless
 *      of N; otherwise you are changing context per slot and concurrency at the
 *      same time and the comparison is confounded).
 *
 * Because --parallel is a server START-UP flag, it cannot be changed from Java
 * while the server runs. The practical procedure is therefore:
 *
 *   for N in {2,3,4}:
 *     (a) (re)start llama-server with --parallel N --ctx-size (N x 12288)
 *     (b) run this harness restricted to the single value N
 *         (pass N as a program argument, e.g. "3")
 *     (c) record the documents/min it prints
 *   pick the N with the best documents/min that stays within the ~15 GB VRAM
 *   ceiling (watch: nvidia-smi --query-gpu=utilization.gpu,memory.used
 *   --format=csv -l 2).
 *
 * If you pass NO argument, the harness will sweep SWEEP_VALUES in one run. Only
 * do that if the server was started with --parallel >= max(SWEEP_VALUES); the
 * lower settings will still be measured correctly (they simply under-use the
 * available slots), but the comparison is less clean than restarting the server
 * to match each N. For the most accurate result, prefer the per-N procedure above.
 *
 * Failed documents are counted and reported but do not abort the sweep. If any
 * documents fail at a setting, its docs/min is not directly comparable with the
 * other settings (fewer documents were completed), so fix the failures first.
 */
public class VascularThroughputHarness {

    // -------------------------------------------------------------------------
    // HARNESS PARAMETERS (TEST THESE)
    // -------------------------------------------------------------------------
    // THREADS values to sweep when no command-line argument is supplied.
    private static final int[] SWEEP_VALUES = { 2, 3, 4 };

    // Number of documents from the corpus to use for each measured run. Keep
    // this modest (enough to average out per-document variance, small enough to
    // run quickly). All runs use the SAME sample for a fair comparison. The
    // corpus listing is sorted, so the sample is identical between program runs.
    private static final int SAMPLE_SIZE = 20;

    // One warm-up document is processed and discarded before timing each setting
    // so that one-time costs (prompt-cache priming of the system prefix, JIT
    // warm-up, first-request connection setup) do not skew the measurement.
    private static final boolean WARM_UP = true;

    private static final String SERVER_URL = "http://localhost:8080";
    private static final String MODEL_NAME = "local-model";

    public static void main(String[] args) throws Exception {

        String corpusDir;
        try (Scanner scanner = new Scanner(System.in)) {
            System.out.print("Enter the corpus directory path to sample from: ");
            corpusDir = scanner.nextLine().trim();
        }

        // Determine which THREADS values to measure.
        int[] sweep;
        if (args != null && args.length > 0) {
            sweep = new int[args.length];
            for (int i = 0; i < args.length; i++) {
                sweep[i] = Integer.parseInt(args[i].trim());
            }
        } else {
            sweep = SWEEP_VALUES;
        }

        // Load and sample the corpus once; reuse the identical sample for every
        // setting so differences reflect concurrency, not document selection.
        List<Path> allDocs = MainVascular.getFilePaths(corpusDir, false);
        if (allDocs.isEmpty()) {
            System.err.println("No documents found in: " + corpusDir);
            return;
        }
        int sampleSize = Math.min(SAMPLE_SIZE, allDocs.size());
        List<Path> sample = new ArrayList<>(allDocs.subList(0, sampleSize));
        Path warmUpDoc = sample.get(0);

        LlamaVascularExtractor extractor =
                new LlamaVascularExtractor(SERVER_URL, MODEL_NAME, MainVascular.SYSTEM_PROMPT);

        System.out.println("Throughput sweep");
        System.out.println("  Sample size:       " + sampleSize + " documents");
        System.out.println("  Warm-up per run:   " + WARM_UP);
        System.out.println("  THREADS values:    " + java.util.Arrays.toString(sweep));
        System.out.println("  REMINDER: start llama-server with --parallel equal to the");
        System.out.println("            THREADS value being measured and --ctx-size = N x 12288.");
        System.out.println();

        List<String> summaryRows = new ArrayList<>();
        summaryRows.add(String.format("%-10s %-10s %-10s %-16s %-14s",
                "THREADS", "ok", "failed", "elapsed(min)", "docs/min"));

        for (int threads : sweep) {
            System.out.println("=== Measuring THREADS = " + threads
                    + "  (ensure server --parallel = " + threads + ") ===");

            if (WARM_UP) {
                try {
                    extractor.analyzeDocument(warmUpDoc); // discarded
                } catch (Exception e) {
                    System.err.println("Warm-up failed (continuing): " + e.getMessage());
                }
            }

            AtomicInteger failed = new AtomicInteger(0);

            long begin = System.currentTimeMillis();
            List<VascularResult> results = extractor.analyzeDocumentsParallel(
                    sample, threads,
                    r -> { /* result discarded; only timing is measured */ },
                    (path, e) -> {
                        failed.incrementAndGet();
                        System.err.println("  FAILED " + path.getFileName() + ": " + e.getMessage());
                    });
            long elapsedMs = System.currentTimeMillis() - begin;

            double elapsedMin = elapsedMs / 60000.0;
            double docsPerMin = results.size() / elapsedMin;

            System.out.printf("  -> %d ok, %d failed in %.3f min  =  %.2f docs/min%n%n",
                    results.size(), failed.get(), elapsedMin, docsPerMin);

            summaryRows.add(String.format("%-10d %-10d %-10d %-16.3f %-14.2f",
                    threads, results.size(), failed.get(), elapsedMin, docsPerMin));
        }

        System.out.println("================ SUMMARY ================");
        summaryRows.forEach(System.out::println);
        System.out.println("========================================");
        System.out.println("Pick the THREADS value with the highest docs/min that keeps");
        System.out.println("memory.used safely under the ~15 GB VRAM ceiling under load.");
        System.out.println("Rows with failed > 0 are not directly comparable; fix failures first.");
    }
}
