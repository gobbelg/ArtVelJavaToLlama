package core;

import java.nio.file.Path;
import java.nio.file.Paths;
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
 * The harness starts and stops llama-server itself (LlamaServerManager) for each
 * value N it measures, using LlamaServerConfig.vaDefaults().withSlots(N, 12288):
 *   - client THREADS = N = server --parallel (extra threads would only queue),
 *   - --ctx-size = N x 12288, so every slot keeps 12288 tokens regardless of N
 *     (otherwise context per slot and concurrency change together and the
 *     comparison is confounded).
 * Watch VRAM while it runs: nvidia-smi --query-gpu=utilization.gpu,memory.used
 * --format=csv -l 2   (keep it under the ~15 GB ceiling).
 *
 * Pass values as program arguments (e.g. "3") to measure only those; with no
 * arguments SWEEP_VALUES is used. The server must NOT already be running on the
 * configured port (the manager refuses to start in that case).
 *
 * Documents longer than the slot budget are skipped (reported separately), so
 * every setting is measured on the same documents only if none are skipped.
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

    private static final String MODEL_NAME = "local-model";

    // Tokens per slot held constant across the sweep.
    private static final int SLOT_SIZE = 12288;

    public static void main(String[] args) throws Exception {

        // Not closed on purpose: closing a Scanner also closes System.in.
        Scanner scanner = new Scanner(System.in);
        System.out.print("Enter the corpus directory path to sample from: ");
        String corpusDir = scanner.nextLine().trim();

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

        System.out.println("Throughput sweep");
        System.out.println("  Sample size:       " + sampleSize + " documents");
        System.out.println("  Warm-up per run:   " + WARM_UP);
        System.out.println("  THREADS values:    " + java.util.Arrays.toString(sweep));
        System.out.println("  The server is started automatically for each value (--parallel N,");
        System.out.println("  --ctx-size = N x " + SLOT_SIZE + ").");
        System.out.println();

        List<String> summaryRows = new ArrayList<>();
        summaryRows.add(String.format("%-10s %-10s %-10s %-16s %-14s",
                "THREADS", "ok", "failed", "elapsed(min)", "docs/min"));
        Path serverLog = Paths.get(corpusDir).toAbsolutePath().resolve("harness_server.log");

        for (int threads : sweep) {
            LlamaServerConfig config = LlamaServerConfig.vaDefaults().withSlots(threads, SLOT_SIZE);
            System.out.println("=== Measuring THREADS = " + threads + "  (" + config.describe() + ") ===");

            try (LlamaServerManager server = new LlamaServerManager(config, serverLog)) {
                server.start();

                LlamaVascularExtractor extractor = new LlamaVascularExtractor(
                        server.baseUrl(), MODEL_NAME, MainVascular.SYSTEM_PROMPT, server.slotSize());

                if (WARM_UP) {
                    try {
                        extractor.analyzeDocument(warmUpDoc); // discarded
                    } catch (Exception e) {
                        System.err.println("Warm-up failed (continuing): " + e.getMessage());
                    }
                }

                AtomicInteger failed = new AtomicInteger(0);
                AtomicInteger skipped = new AtomicInteger(0);

                long begin = System.currentTimeMillis();
                List<VascularResult> results = extractor.analyzeDocumentsParallel(
                        sample, threads,
                        r -> { /* result discarded; only timing is measured */ },
                        (path, e) -> {
                            if (e instanceof ContextBudget.DocumentTooLongException) {
                                skipped.incrementAndGet();
                                System.out.println("  SKIPPED (too long) " + path.getFileName());
                            } else {
                                failed.incrementAndGet();
                                System.err.println("  FAILED " + path.getFileName() + ": " + e.getMessage());
                            }
                        });
                long elapsedMs = System.currentTimeMillis() - begin;

                double elapsedMin = elapsedMs / 60000.0;
                double docsPerMin = results.size() / elapsedMin;

                System.out.printf("  -> %d ok, %d failed, %d skipped (too long) in %.3f min  =  %.2f docs/min%n%n",
                        results.size(), failed.get(), skipped.get(), elapsedMin, docsPerMin);
                System.out.println("  " + extractor.usageSummary());

                summaryRows.add(String.format("%-10d %-10d %-10d %-16.3f %-14.2f",
                        threads, results.size(), failed.get(), elapsedMin, docsPerMin));
            } catch (Exception e) {
                System.err.println("  Could not measure THREADS = " + threads + ": " + e.getMessage());
                summaryRows.add(String.format("%-10d %-10s", threads, "server error"));
            }
        }

        System.out.println("================ SUMMARY ================");
        summaryRows.forEach(System.out::println);
        System.out.println("========================================");
        System.out.println("Pick the THREADS value with the highest docs/min that keeps");
        System.out.println("memory.used safely under the ~15 GB VRAM ceiling under load.");
        System.out.println("Rows with failed > 0 are not directly comparable; fix failures first.");
    }
}
