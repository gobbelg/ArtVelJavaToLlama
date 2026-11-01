package core;

import config.AppSettings;
import extract.ContextBudget;
import extract.LlamaVascularExtractor;
import extract.VascularPrompt;
import output.ResumeSupport;
import output.VascularTsvWriter;
import server.LlamaServerConfig;
import server.LlamaServerManager;
import telemetry.DocumentTelemetry;
import telemetry.RunRecorder;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ArrayNode;

import model.VascularModels.VascularResult;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Scanner;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class MainVascular {

	// Everything that differs between computers (corpus directory, document
	// extension, server paths, --ctx-size, threads, default parallel, second-pass
	// behavior, HTTP timeouts) comes from the settings file chosen at start-up;
	// see AppSettings. It is loaded at the top of main().
	private static AppSettings settings;

	// -------------------------------------------------------------------------
	// SERVER AND PASS SETTINGS
	// -------------------------------------------------------------------------
	// This program starts and stops llama-server itself (LlamaServerManager),
	// using the server.* values of the settings file, which reproduce the command
	// line you used to launch it by hand. Edit paths and flags there.
	//
	// For every pass, client threads = server slots (--parallel). More threads
	// than slots only queue on the server; fewer leave slots idle.
	//
	// Slot size = --ctx-size / --parallel. For example, with --ctx-size 49152:
	//     --parallel 4 -> 12288 tokens per slot   (first pass)
	//     --parallel 1 -> 49152 tokens per slot   (second pass)
	// The KV-cache (VRAM) cost is the same either way; only concurrency changes.
	//
	// Reports too long for the first-pass slot size (see ContextBudget) are
	// skipped, not failed. If run.auto-second-pass is true, the server is restarted
	// with run.second-pass-parallel slots and those reports are processed then.
	// Anything still too long is written to <output>_too_long.txt.
	// -------------------------------------------------------------------------

	// The system prompt lives in extract/VascularPrompt (VascularPrompt.SYSTEM_PROMPT).


	// -------------------------------------------------------------------------
	// Main
	// -------------------------------------------------------------------------

	public static void main(String[] args) throws Exception {
		ObjectMapper mapper = new ObjectMapper();

		// One Scanner for every console prompt. It is deliberately NOT closed:
		// closing a Scanner on System.in also closes System.in itself.
		Scanner scanner = new Scanner(System.in);

		// Settings for this computer: -Dvascular.config=..., VASCULAR_CONFIG, or a prompt.
		try {
			settings = AppSettings.load(scanner);
		} catch (IOException | IllegalArgumentException e) {
			System.err.println("Unable to load the settings file: " + e.getLocalizedMessage());
			System.exit(-1);
			return;
		}
		System.out.println(settings.describe());
		System.out.println();

		// EXPERIMENT MODE (an experiment file was given): no console prompts. A fresh run
		// folder is created under output.base-directory (see RunRecorder) and the documents
		// come from experiment.document-list or corpus.directory. Otherwise the original
		// interactive behavior is unchanged.
		final boolean experimentMode = settings.experimentMode();
		RunRecorder recorder = null;
		Path outputFile;
		List<Path> documentPaths;
		int firstPassParallel;
		if (experimentMode) {
			try {
				documentPaths = settings.documentList() != null
						? readDocumentList(settings.documentList(), settings.documentList().toAbsolutePath().getParent())
						: getFilePaths(settings.corpusDirectory().toString(), false, settings.documentExtension());
				if (documentPaths.isEmpty()) {
					throw new IllegalArgumentException("No " + settings.documentExtension() + " documents found");
				}
				recorder = RunRecorder.create(settings, effectivePrompt(), VascularPrompt.SYSTEM_PROMPT, documentPaths);
			} catch (IOException | IllegalArgumentException e) {
				System.err.println("Unable to set up the experiment run: " + e.getLocalizedMessage());
				e.printStackTrace();
				System.exit(-1);
				return;
			}
			outputFile = recorder.resultsFile();
			firstPassParallel = settings.defaultParallel();
			System.out.println("Experiment run " + recorder.runId() + " (" + documentPaths.size()
					+ " documents)\nRun folder: " + recorder.runDir().toAbsolutePath() + "\n");
		} else {
			outputFile = getOutputFile(scanner);
			try {
				documentPaths = getDocumentPaths(scanner);
			} catch (IOException | IllegalArgumentException e) {
				System.err.println("Unable to load the documents to process: " + e.getLocalizedMessage());
				e.printStackTrace();
				System.exit(-1);
				return;
			}
			firstPassParallel = getParallel(scanner);
		}

		// --- Resume: skip documents already recorded in an existing output file ---
		// Only successfully processed documents are in that file, so documents that
		// failed or were too long on an earlier run are picked up again. Re-running
		// with the SAME output file name (answer N to the timestamp prompt) therefore
		// never duplicates records.
		if (Files.exists(outputFile) && Files.size(outputFile) > 0) {
			if (askYesNo(scanner, "The output file already contains results. Skip documents already "
					+ "recorded in it (Y/N) [Y]: ", true)) {
				Set<String> alreadyDone = ResumeSupport.loadProcessedPaths(outputFile, mapper, System.out::println);
				int before = documentPaths.size();
				documentPaths = documentPaths.stream()
						.filter(p -> !alreadyDone.contains(ResumeSupport.normalize(p.toString())))
						.collect(Collectors.toList());
				System.out.println("Skipping " + (before - documentPaths.size()) + " of " + before
						+ " documents already in the output file; " + documentPaths.size() + " to process.");
				if (documentPaths.isEmpty()) {
					System.out.println("Nothing to do.");
					return;
				}
			}
		}

		// --- Set up synchronized writer (append mode) ---
		PrintWriter resultWriter = new PrintWriter(new BufferedWriter(
				new OutputStreamWriter(new FileOutputStream(outputFile.toFile(), true), StandardCharsets.UTF_8)));

		// --- Set up synchronized logger (append mode) ---
		Path logFile = getLogFile(outputFile);
		PrintWriter logger = new PrintWriter(new BufferedWriter(
				new OutputStreamWriter(new FileOutputStream(logFile.toFile(), true), StandardCharsets.UTF_8)));

		// llama-server's own console output is appended to this file.
		Path serverLogFile = experimentMode ? recorder.serverLogFile() : getSiblingFile(outputFile, "_server.log");

		// --- Set up the three tab-delimited result files (append mode) ---
		// <output>_ICA_CCA_ratio.tsv, <output>_flow_velocities.tsv, <output>_stenosis.tsv
		// In experiment mode every file also gets a run_id column (and documents.tsv gets timing and
		// token columns).
		VascularTsvWriter tsvWriter = new VascularTsvWriter(outputFile, experimentMode ? recorder.runId() : null);
		tsvWriter.getFiles().forEach(f -> System.out.println("Tab-delimited results: " + f.toAbsolutePath()));

		RunContext ctx = new RunContext(mapper, resultWriter, logger, tsvWriter, recorder);

		String runStatus = "ABORTED";
		int tooLongCount = 0;
		int notProcessedCount = 0;
		try {
			long beginTime = System.currentTimeMillis();
			ctx.log("Begin processing: " + beginTime + " (" + documentPaths.size() + " documents)");

			// Documents still to do. After each pass this becomes the list of
			// documents that were too long for that pass's slot size.
			List<Path> pending = documentPaths;
			List<Path> failedDocuments = new ArrayList<>();
			int succeeded = 0;
			int parallel = firstPassParallel;
			int pass = 1;
			boolean aborted = false;

			try {
				while (!pending.isEmpty()) {
					PassResult passResult = runPass(ctx, pass, parallel, pending, serverLogFile);
					succeeded += passResult.succeeded;
					failedDocuments.addAll(passResult.failed);
					pending = passResult.tooLong;

					if (pending.isEmpty() || !settings.autoSecondPass() || parallel <= settings.secondPassParallel()) {
						break;
					}
					ctx.log(pending.size() + " document(s) were too long for the current slot size; "
							+ "restarting the server with --parallel " + settings.secondPassParallel() + " for a second pass.");
					parallel = settings.secondPassParallel();
					pass++;
				}
			} catch (Exception e) {
				// Typically the server could not be started. Documents of a pass
				// that was interrupted may already have results; check the outputs.
				aborted = true;
				ctx.log("RUN ABORTED during pass " + pass + ": " + e.getMessage());
				e.printStackTrace();
			}

			long endTime = System.currentTimeMillis();
			ctx.log("End processing: " + endTime);
			ctx.log(String.format("Total processing time: %.2f min", (endTime - beginTime) / 60000.0));

			List<Path> tooLongDocuments = aborted ? new ArrayList<>() : pending;
			List<Path> notProcessed = aborted ? pending : new ArrayList<>();
			runStatus = aborted ? "ABORTED" : "COMPLETED";
			tooLongCount = tooLongDocuments.size();
			notProcessedCount = notProcessed.size();

			ctx.log("Succeeded: " + succeeded + "  Failed: " + failedDocuments.size()
					+ "  Too long for the largest slot size used: " + tooLongDocuments.size()
					+ (aborted ? "  Not processed (run aborted): " + notProcessed.size() : ""));

			// Documents-file rows for documents that never produced a result.
			for (Path p : tooLongDocuments) {
				tsvWriter.writeStatus(p.toString(), "TOO_LONG", pass, "",
						"Longer than the largest slot size used in this run");
			}
			for (Path p : notProcessed) {
				tsvWriter.writeStatus(p.toString(), "NOT_PROCESSED", pass, "",
						"The run aborted before this document was reached");
			}

			writeList(ctx, getSiblingFile(outputFile, "_failed.txt"), failedDocuments,
					"Failed documents");
			writeList(ctx, getSiblingFile(outputFile, "_too_long.txt"), tooLongDocuments,
					"Documents too long even for the largest slot size used (re-run them with a "
							+ "smaller parallel value, or shorten them)");
			writeList(ctx, getSiblingFile(outputFile, "_not_processed.txt"), notProcessed,
					"Documents not processed because the run aborted (may include documents of the "
							+ "interrupted pass that already have results)");
		} finally {
			resultWriter.close();
			tsvWriter.close();
			logger.close();
			if (recorder != null) {
				recorder.finish(runStatus, tooLongCount, notProcessedCount);
				System.out.println("Run folder: " + recorder.runDir().toAbsolutePath());
			}
		}
	}

	/** The system prompt in effect: prompt.file from the settings, or the one built into this class. */
	static String effectivePrompt() {
		return settings.effectivePrompt(VascularPrompt.SYSTEM_PROMPT);
	}

	// -------------------------------------------------------------------------
	// One pass: start the server with the given slot count, process documents,
	// stop the server. Too-long documents are returned, not treated as failures.
	// -------------------------------------------------------------------------

	private static PassResult runPass(RunContext ctx, int pass, int parallel, List<Path> documents,
			Path serverLogFile) throws Exception {

		// --ctx-size stays at the configured total; fewer slots means larger slots.
		LlamaServerConfig config = settings.serverConfig().withParallel(parallel);
		final int totalDocs = documents.size();
		ctx.log("=== Pass " + pass + ": " + totalDocs + " documents; " + config.describe()
				+ "; client threads = " + parallel + " ===");

		List<Path> failed = new ArrayList<>();
		List<Path> tooLong = new ArrayList<>();
		AtomicInteger documentNumber = new AtomicInteger(0);
		AtomicInteger succeeded = new AtomicInteger(0);

		try (LlamaServerManager server = new LlamaServerManager(config, serverLogFile)) {
			final long startNanos = System.nanoTime();
			server.start();
			final double serverStartSeconds = (System.nanoTime() - startNanos) / 1e9;
			ctx.log(String.format("Server ready after %.1f s.", serverStartSeconds));
			final int slotSize = server.slotSize();

			LlamaVascularExtractor extractor = new LlamaVascularExtractor(server.baseUrl(), settings.modelName(),
					effectivePrompt(), slotSize, settings.requestTimeout(), settings.connectTimeout(),
					settings.extractionOptions());
			ctx.log(extractor.getBudget().describe());

			final long passBegin = System.currentTimeMillis();

			extractor.analyzeDocumentsParallel(documents, parallel, result -> {
				String json;
				try {
					json = ctx.mapper.writerWithDefaultPrettyPrinter().writeValueAsString(result);
				} catch (Exception e) {
					// Counts as a failure so the document appears in the re-run list.
					throw new RuntimeException("Failed to serialize result for: "
							+ result.documentPathString() + " — " + e.getMessage(), e);
				}
				// Timing/tokens/raw text for this document (always taken so nothing accumulates).
				final DocumentTelemetry telemetry = extractor.takeTelemetry(result.documentPathString());
				final boolean noFindings = result.ICA_CCA_ratio().isEmpty() && result.flow_velocities().isEmpty()
						&& result.stenosis().isEmpty();
				synchronized (ctx.writeLock) {
					int current = documentNumber.incrementAndGet();
					succeeded.incrementAndGet();
					recordDocument(ctx, noFindings ? "SUCCESS_NO_FINDINGS" : "SUCCESS",
							Paths.get(result.documentPathString()), telemetry, pass);

					ctx.resultWriter.println(json);
					ctx.resultWriter.println(); // blank line between records
					ctx.resultWriter.flush();

					// Tab-delimited files. A failure here must not mark the document
					// as failed (its JSON is already written), so log and continue.
					try {
						ctx.tsvWriter.write(result, pass, extractor.takeNotes(result.documentPathString()), telemetry);
					} catch (Exception tsvException) {
						String msg = "Failed to write tab-delimited rows for "
								+ result.documentPathString() + ": " + tsvException.getMessage();
						System.err.println(msg);
						ctx.logger.println(msg);
						ctx.logger.flush();
					}

					StringBuilder sb = new StringBuilder();
					sb.append("Result written for document ").append(current).append(" of ")
							.append(totalDocs).append(" documents (pass ").append(pass).append(")\n");
					sb.append("\tFileName: ").append(Paths.get(result.documentPathString()).getFileName()).append("\n");
					sb.append(String.format("Estimated remaining processing time: %.2f min%n",
							estimateRemainingProcessingTime(current, totalDocs, passBegin)));

					System.out.println(sb.toString());
					ctx.logger.println(sb.toString());
					ctx.logger.flush();
				}
			}, (path, e) -> {
				final DocumentTelemetry telemetry = extractor.takeTelemetry(path.toString());
				synchronized (ctx.writeLock) {
					int current = documentNumber.incrementAndGet();
					boolean isTooLong = e instanceof ContextBudget.DocumentTooLongException;
					if (isTooLong) {
						tooLong.add(path);
						extractor.takeNotes(path.toString());
					} else {
						failed.add(path);
						ctx.tsvWriter.writeStatus(path.toString(), "FAILED", pass,
								extractor.takeNotes(path.toString()), e.getMessage(), telemetry);
						recordDocument(ctx, "FAILED", path, telemetry, pass);
					}

					StringBuilder sb = new StringBuilder();
					sb.append(isTooLong ? "SKIPPED (too long for slot size " + slotSize + ") document "
							: "FAILED document ").append(current).append(" of ").append(totalDocs)
							.append(" documents (pass ").append(pass).append(")\n");
					sb.append("\tFileName: ").append(path.getFileName()).append("\n");
					sb.append("\t").append(isTooLong ? "Reason: " : "Error: ").append(e.getMessage()).append("\n");
					sb.append(String.format("Estimated remaining processing time: %.2f min%n",
							estimateRemainingProcessingTime(current, totalDocs, passBegin)));

					(isTooLong ? System.out : System.err).println(sb.toString());
					ctx.logger.println(sb.toString());
					ctx.logger.flush();
				}
			});

			final double processingSeconds = (System.currentTimeMillis() - passBegin) / 1000.0;
			ctx.log(String.format("Pass %d finished in %.2f min: %d succeeded, %d too long for slot size %d, %d failed.",
					pass, processingSeconds / 60.0, succeeded.get(), tooLong.size(), slotSize, failed.size()));
			if (ctx.recorder != null) {
				ctx.recorder.recordPass(pass, parallel, slotSize, serverStartSeconds, processingSeconds,
						totalDocs, succeeded.get(), tooLong.size(), failed.size());
			}
			ctx.log(extractor.usageSummary());
		} // the server is stopped here, before any next pass restarts it

		return new PassResult(succeeded.get(), failed, tooLong);
	}

	// Passes the document's status, timing and raw responses to the run recorder (experiment mode).
	// Never throws: a recording problem must not mark a document as failed.
	private static void recordDocument(RunContext ctx, String status, Path path, DocumentTelemetry telemetry,
			int pass) {
		if (ctx.recorder == null) {
			return;
		}
		try {
			ctx.recorder.documentDone(status, telemetry);
			ctx.recorder.saveRaw(path, telemetry, pass);
		} catch (Exception e) {
			System.err.println("Unable to record telemetry for " + path + ": " + e.getMessage());
		}
	}

	// Shared state for a run. All console/log output and result writing goes
	// through writeLock so lines from concurrent documents do not interleave.
	private static final class RunContext {
		final ObjectMapper mapper;
		final PrintWriter resultWriter;
		final PrintWriter logger;
		final VascularTsvWriter tsvWriter;
		final RunRecorder recorder;   // null unless in experiment mode
		final Object writeLock = new Object();

		RunContext(ObjectMapper mapper, PrintWriter resultWriter, PrintWriter logger,
				VascularTsvWriter tsvWriter, RunRecorder recorder) {
			this.recorder = recorder;
			this.mapper = mapper;
			this.resultWriter = resultWriter;
			this.logger = logger;
			this.tsvWriter = tsvWriter;
		}

		void log(String message) {
			synchronized (writeLock) {
				System.out.println(message);
				logger.println(message);
				logger.flush();
			}
		}
	}

	private static final class PassResult {
		final int succeeded;
		final List<Path> failed;
		final List<Path> tooLong;

		PassResult(int succeeded, List<Path> failed, List<Path> tooLong) {
			this.succeeded = succeeded;
			this.failed = failed;
			this.tooLong = tooLong;
		}
	}

	private static Path getLogFile(Path outputPath) {

		String outputFileName = outputPath.toFile().getName();
		int extIndex = outputFileName.indexOf(".");
		String logFileName = extIndex > 0 ? outputFileName.substring(0, extIndex) + ".log" : "logFile.log";

		return outputPath.getParent().resolve(logFileName);
	}

	// e.g. results_261026_101500.json + "_too_long.txt" -> results_261026_101500_too_long.txt
	private static Path getSiblingFile(Path outputPath, String suffix) {

		String outputFileName = outputPath.getFileName().toString();
		int extIndex = outputFileName.lastIndexOf(".");
		String base = extIndex > 0 ? outputFileName.substring(0, extIndex) : outputFileName;

		return outputPath.toAbsolutePath().getParent().resolve(base + suffix);
	}

	private static void writeList(RunContext ctx, Path file, List<Path> paths, String description) {
		if (paths.isEmpty()) {
			return;
		}
		try {
			Files.write(file, paths.stream().map(Path::toString).collect(Collectors.toList()),
					StandardCharsets.UTF_8);
			ctx.log(description + ": " + paths.size() + " document(s); list written to "
					+ file.toAbsolutePath());
		} catch (IOException e) {
			ctx.log("Unable to write " + file.toAbsolutePath() + ": " + e.getMessage());
		}
	}

	private static double estimateRemainingProcessingTime(int documentNumber, int totalDocs, long beginTime) {

		long elapsedTime = Math.max(1L, System.currentTimeMillis() - beginTime);
		int remainingDocuments = totalDocs - documentNumber;

		double processingRate = (1.0 * documentNumber) / elapsedTime;
		double estimatedProcessingTimeLeft = remainingDocuments / processingRate;

		return estimatedProcessingTimeLeft / 60000;
	}

	// -------------------------------------------------------------------------
	// Console prompts (all share the one Scanner created in main)
	// -------------------------------------------------------------------------

	private static Path getOutputFile(Scanner scanner) throws IOException {

		// --- Prompt user for output file location ---
		System.out.print("Enter the output directory path: ");
		String outputDirInput = scanner.nextLine().trim();

		System.out.print("Enter the output file name (e.g. results): ");
		String outputFileName = scanner.nextLine().trim();

		String addTimeStamp = "";
		while (!addTimeStamp.toLowerCase().startsWith("y") && !addTimeStamp.toLowerCase().startsWith("n")) {
			System.out.print("Add timestamp to filename, (Y)es or (N)o: ");
			addTimeStamp = scanner.nextLine().trim();
		}
		if (addTimeStamp.toLowerCase().startsWith("y")) {
			outputFileName += "_" + getTimeStamp();
		}
		outputFileName += ".json";

		Path outputDir = Paths.get(outputDirInput);
		Path outputFile = outputDir.resolve(outputFileName);

		// Create output directory if it does not exist
		if (!Files.exists(outputDir)) {
			Files.createDirectories(outputDir);
			System.out.println("Created output directory: " + outputDir.toAbsolutePath());
		}

		// Warn if the file already exists — results will be appended
		if (Files.exists(outputFile)) {
			System.out.println("Warning: output file already exists. Results will be appended: "
					+ outputFile.toAbsolutePath());
		} else {
			System.out.println("Results will be written to: " + outputFile.toAbsolutePath());
		}
		return outputFile;
	}

	// Either every document in corpus.directory (settings file), or the paths listed in a text
	// file (one per line; blank lines and lines starting with # are ignored). The
	// _too_long.txt and _failed.txt files written by a run are valid list files.
	private static List<Path> getDocumentPaths(Scanner scanner) throws IOException {

		Path corpus = settings.corpusDirectory();
		if (corpus == null) {
			System.out.print("Enter the corpus directory (no corpus.directory in the settings file): ");
			corpus = Paths.get(stripQuotes(scanner.nextLine().trim()));
			System.out.print("Enter the path of a document list file (one path per line), or press Enter to "
					+ "process every " + settings.documentExtension() + " file in that directory: ");
		} else {
			System.out.print("Enter the path of a document list file (one path per line), or press Enter to "
					+ "process every " + settings.documentExtension() + " file in " + corpus + ": ");
		}
		String input = stripQuotes(scanner.nextLine().trim());

		if (input.isEmpty()) {
			return getFilePaths(corpus.toString(), false, settings.documentExtension());
		}

		Path listFile = Paths.get(input);
		if (!Files.isRegularFile(listFile)) {
			throw new IllegalArgumentException("List file not found: " + listFile.toAbsolutePath());
		}

		List<Path> paths = readDocumentList(listFile, null);
		System.out.println("Loaded " + paths.size() + " document paths from " + listFile.toAbsolutePath());
		return paths;
	}

	// Reads a document list file (one path per line; blank lines and # comments ignored).
	// A relative entry is resolved against baseDir when one is given.
	private static List<Path> readDocumentList(Path listFile, Path baseDir) throws IOException {
		List<Path> paths = new ArrayList<>();
		for (String line : Files.readAllLines(listFile, StandardCharsets.UTF_8)) {
			String trimmed = line.replace("\uFEFF", "").trim();
			if (!trimmed.isEmpty() && !trimmed.startsWith("#")) {
				Path p = Paths.get(stripQuotes(trimmed));
				if (!p.isAbsolute() && baseDir != null) {
					p = baseDir.resolve(p).normalize();
				}
				paths.add(p);
			}
		}
		if (paths.isEmpty()) {
			throw new IllegalArgumentException("The list file contains no document paths: "
					+ listFile.toAbsolutePath());
		}
		return paths;
	}

	private static boolean askYesNo(Scanner scanner, String prompt, boolean defaultValue) {
		while (true) {
			System.out.print(prompt);
			String input = scanner.nextLine().trim().toLowerCase();
			if (input.isEmpty()) {
				return defaultValue;
			}
			if (input.startsWith("y")) {
				return true;
			}
			if (input.startsWith("n")) {
				return false;
			}
		}
	}

	private static int getParallel(Scanner scanner) {
		while (true) {
			System.out.print("Number of parallel server slots for the first pass [" + settings.defaultParallel()
					+ "] (slot size = " + settings.serverConfig().ctxSize()
					+ " / slots; use 1 to run long reports): ");
			String input = scanner.nextLine().trim();
			if (input.isEmpty()) {
				return settings.defaultParallel();
			}
			try {
				int n = Integer.parseInt(input);
				if (n >= 1) {
					return n;
				}
			} catch (NumberFormatException ignored) {
				// fall through to the message below
			}
			System.out.println("Enter a whole number of 1 or more.");
		}
	}

	private static String stripQuotes(String s) {
		if (s.length() >= 2 && s.startsWith("\"") && s.endsWith("\"")) {
			return s.substring(1, s.length() - 1);
		}
		return s;
	}

	// -------------------------------------------------------------------------
	// JSON array conversion
	// -------------------------------------------------------------------------

	private static void convertToJsonArray(Path inputPath, Path outputPath) throws IOException {
		ObjectMapper mapper = new ObjectMapper();
		ArrayNode array = mapper.createArrayNode();

		try (InputStream is = new BufferedInputStream(Files.newInputStream(inputPath))) {
			var iterator = mapper.readerFor(JsonNode.class).readValues(is);
			int count = 0;
			while (iterator.hasNext()) {
				array.add((JsonNode) iterator.next());
				count++;
			}
			System.out.println("Read " + count + " records from: " + inputPath.getFileName());
		}

		try (OutputStream os = new BufferedOutputStream(
				Files.newOutputStream(outputPath, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING))) {
			mapper.writerWithDefaultPrettyPrinter().writeValue(os, array);
		}

		System.out.println("JSON array written to: " + outputPath.toAbsolutePath());
	}

	/**
	 * Reads an append-mode JSON file (a sequence of JSON objects separated by blank
	 * lines) and writes a single valid JSON array to a new file.
	 *
	 * @param outputPath Path to the append-mode result file produced by the
	 *                   extractor.
	 * @throws IOException If either file cannot be read or written.
	 */
	public static void convertToJsonArray(Path outputPath) throws IOException {

		// --- Convert the completed result file to a JSON array ---
		// Derive array output path by inserting "_array" before the file extension.
		// e.g. results.json -> results_array.json
		String originalName = outputPath.getFileName().toString();

		int dot = originalName.lastIndexOf('.');
		String arrayName = (dot >= 0) ? originalName.substring(0, dot) + "_array" + originalName.substring(dot)
				: originalName + "_array";
		Path arrayOutputFile = outputPath.getParent().resolve(arrayName);

		System.out.println("Converting result file to JSON array...");
		try {
			convertToJsonArray(outputPath, arrayOutputFile);
		} catch (IOException e) {
			System.err.println("JSON array conversion failed: " + e.getMessage());
		}
	}

	// -------------------------------------------------------------------------
	// File utilities
	// -------------------------------------------------------------------------

	/**
	 * Returns the document file paths in a directory, filtered to
	 * documentExtension (corpus.extension in the settings file) and sorted by path. Sorting makes the order (and
	 * therefore the throughput harness's sample) deterministic; Files.walk order
	 * is otherwise filesystem-dependent.
	 *
	 * @param dirPath   Absolute path to the directory.
	 * @param recursive If true, descends into subdirectories.
	 * @return Sorted list of matching file paths.
	 * @throws IOException              If the directory cannot be read.
	 * @throws IllegalArgumentException If the path is invalid or not a directory.
	 */
	public static List<Path> getFilePaths(String dirPath, boolean recursive) throws IOException {
		return getFilePaths(dirPath, recursive, ".txt");
	}

	public static List<Path> getFilePaths(String dirPath, boolean recursive, String documentExtension)
			throws IOException {
		Path directory = Paths.get(dirPath);

		if (!Files.exists(directory)) {
			throw new IllegalArgumentException("Path does not exist: " + dirPath);
		}
		if (!Files.isDirectory(directory)) {
			throw new IllegalArgumentException("Path is not a directory: " + dirPath);
		}

		int depth = recursive ? Integer.MAX_VALUE : 1;
		String ext = documentExtension.toLowerCase();

		try (Stream<Path> paths = Files.walk(directory, depth)) {
			return paths.filter(Files::isRegularFile)
					.filter(p -> ext.isEmpty() || p.getFileName().toString().toLowerCase().endsWith(ext))
					.sorted()
					.collect(Collectors.toList());
		}
	}

	public static String getTimeStamp() {
		DateFormat dateFormat = new SimpleDateFormat("yyMMdd_HHmmss");
		Date date = new Date();
		return dateFormat.format(date);
	}
}
