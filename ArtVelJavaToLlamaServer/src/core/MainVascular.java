package core;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ArrayNode;

import core.VascularModels.VascularResult;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Scanner;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class MainVascular {

	private static final String CORPUS_DIRECTORY_PATH = "P:/ORD_Nguyen_202206062D/Glenn/DataAnalysis/LLMs/TestCorpus2";

	// Only files with this extension (case-insensitive) are processed. Prevents
	// stray non-report files (Thumbs.db, .log, etc.) in the corpus directory from
	// being sent to the model. Set to "" to disable the filter.
	private static final String DOCUMENT_EXTENSION = ".txt";

	// -------------------------------------------------------------------------
	// THROUGHPUT TUNING PARAMETER (TEST THIS)
	// -------------------------------------------------------------------------
	// THREADS is the number of concurrent in-flight HTTP requests to the
	// llama.cpp server. It MUST equal the server's --parallel value: higher than
	// --parallel just queues requests server-side (no gain); lower leaves the
	// server's KV-cache slots idle.
	//
	// Recommended starting pair (each slot gets --ctx-size / --parallel tokens):
	//
	//   llama-server.exe --host 127.0.0.1 --port 8080 --model "<path>.gguf"
	//       --n-gpu-layers 999 --ctx-size 49152 --parallel 4 --cont-batching
	//       --batch-size 2048 --ubatch-size 512 --flash-attn on --threads 12
	//       --mlock   (12288 tokens per slot)
	//
	//   THREADS = 4   (this constant)
	//
	// The GPU (NVIDIA A2), not the CPU, is the bottleneck. Use
	// VascularThroughputHarnessRevised (RUN_THROUGHPUT_SWEEP) to compare
	// --parallel/THREADS = 2, 3 and 4 (restart the server for each, keeping
	// --ctx-size = N x 12288), and keep the value where aggregate documents/minute
	// stops improving. Watch `nvidia-smi --query-gpu=utilization.gpu,memory.used
	// --format=csv -l 2` under load: if GPU-Util pins near 100%, more slots
	// won't help; if memory.used nears the ~15 GB ceiling, back off.
	// -------------------------------------------------------------------------
	private static final int THREADS = 4;

	// -------------------------------------------------------------------------
	// SYSTEM PROMPT — fixed instructions, schema, and worked example.
	// -------------------------------------------------------------------------
	// Sent once as the Gemma 4 `system` role message. Because it is constant
	// across every document, pairing it with cache_prompt=true on the request
	// lets llama.cpp process this prefix once and reuse its KV cache for all
	// subsequent documents.
	//
	// The schema below is aligned with the XML schema used for manually extracted
	// values and MUST stay in sync with: buildJsonRepairPrompt() and
	// buildResponseFormat() in LlamaVascularExtractorRevised, and the records in
	// VascularModelsRevised.
	//
	// NOTE: literal % characters (e.g. stenosis percentages) are written as a
	// SINGLE % here. Do NOT double the percent signs.
	// -------------------------------------------------------------------------
	static final String SYSTEM_PROMPT = """
			You are a precise medical data extraction assistant. Your sole function \
			is to read vascular ultrasound reports and extract specific quantitative \
			findings into a structured JSON format. You do not interpret, diagnose, \
			or summarize. You extract only what is explicitly stated in the report.

			Rules you must follow without exception:
			- Output only valid JSON. Do not include any explanatory text, preamble, \
			or commentary before or after the JSON.
			- Do not use markdown. Do not wrap output in code fences.
			- If laterality or vessel_location is not present in the report, use the string \
			"InsufficientInformation".
			- Never output an entry whose ratio_value, value, or percentage is not stated in the \
			report. Omit that entry entirely instead of using "InsufficientInformation" for it.
			- If a vessel is mentioned but explicitly stated to have no measurable flow \
			or is described as occluded, use the string "NoMeasurableFlow" as the value. \
			For such an entry use velocity_type "PSV" unless the report specifies EDV.
			- Do not infer or estimate values. Only extract values explicitly stated.
			- Laterality must be recorded for every entry. If laterality is \
			ambiguous or unstated, use "InsufficientInformation".
			- Velocity values must preserve the numeric value exactly as stated in the \
			report. Do not convert units and do not include units in the value.
			- velocity_type must be "PSV" or "EDV" (acronyms for peak systolic velocity \
			and end diastolic velocity, respectively). If the report does not state whether a \
			velocity is PSV or EDV, use "PSV".
			- flow_direction must be "anterograde" or "retrograde". Use "retrograde" only if \
			the report explicitly describes retrograde, reversed, or reverse flow for that \
			measurement. If the flow direction is not stated in the report, use "anterograde".
			- Vessel location (proximal, mid, or distal) must be recorded for every velocity \
			measurement. If the portion of the vessel is not stated, use "InsufficientInformation". \
			If measurements are reported at more than one location on the same vessel, each \
			location must be reported as a separate entry in the array.
			- Stenosis values must preserve the exact percentage or range stated \
			(e.g., "60-69%", "<50%", "70%").
			- Extract ONLY the ICA/CCA ratio, flow velocities, and stenosis. Ignore plaque, \
			intimal thickening, and all other findings.
			- source_text must be copied CHARACTER-FOR-CHARACTER from the report: a short excerpt \
			(roughly 15 to 40 characters, normally within one line) that contains the extracted \
			value exactly as written, plus a few neighboring words such as the vessel name or the \
			PSV/EDV label. Do not paraphrase, reorder, correct, abbreviate, or reformat it, and do \
			not add or remove spaces or punctuation. Choose the excerpt so that it is as specific \
			as possible to this measurement. Every entry must include source_text.
			- Use empty arrays [] for sections with no findings.
			- For unilateral studies, include entries only for the documented side.

			Extract all relevant vascular findings from the carotid ultrasound report \
			provided by the user and return them in the JSON structure specified below.

			Required JSON structure:
			{
			  "ICA_CCA_ratio": [
			    {
			      "laterality": "Left | Right | InsufficientInformation",
			      "source_text": "<short excerpt copied verbatim from the report that contains the ratio value>",
			      "ratio_value": "<numeric value as string>"
			    }
			  ],
			  "flow_velocities": [
			    {
			      "vessel": "ICA | ECA | CCA",
			      "laterality": "Left | Right | InsufficientInformation",
			      "vessel_location": "proximal | mid | distal | InsufficientInformation",
			      "velocity_type": "PSV | EDV",
			      "flow_direction": "anterograde | retrograde",
			      "source_text": "<short excerpt copied verbatim from the report that contains the velocity value>",
			      "value": "<numeric value as string, or NoMeasurableFlow>"
			    }
			  ],
			  "stenosis": [
			    {
			      "vessel": "ICA | ECA | CCA",
			      "laterality": "Left | Right | InsufficientInformation",
			      "source_text": "<short excerpt copied verbatim from the report that contains the percentage>",
			      "percentage": "<exact value or range as stated>"
			    }
			  ]
			}

			Example input:
			<<<
			RIGHT SIDE:
			CCA: PSV 82 cm/s, EDV 24 cm/s. No plaque.
			ICA: Proximal ICA PSV 124 cm/s, EDV 42 cm/s. Distal ICA PSV 98 cm/s, EDV 31 cm/s. ICA/CCA ratio: 1.5. Mild plaque proximal ICA. Stenosis 40-59%.
			ECA: PSV 96 cm/s.
			LEFT SIDE:
			CCA: PSV 78 cm/s, EDV 22 cm/s. Mild intimal thickening mid CCA.
			ICA: PSV 310 cm/s, EDV 115 cm/s. ICA/CCA ratio: 3.97. Dense calcified plaque at bifurcation. Stenosis 70-79%.
			ECA: PSV 110 cm/s with retrograde flow.
			>>>

			Example output:
			{"ICA_CCA_ratio":[{"laterality":"Right","source_text":"ICA/CCA ratio: 1.5","ratio_value":"1.5"},{"laterality":"Left","source_text":"ICA/CCA ratio: 3.97","ratio_value":"3.97"}],"flow_velocities":[{"vessel":"CCA","laterality":"Right","vessel_location":"InsufficientInformation","velocity_type":"PSV","flow_direction":"anterograde","source_text":"CCA: PSV 82 cm/s","value":"82"},{"vessel":"CCA","laterality":"Right","vessel_location":"InsufficientInformation","velocity_type":"EDV","flow_direction":"anterograde","source_text":"PSV 82 cm/s, EDV 24 cm/s","value":"24"},{"vessel":"ICA","laterality":"Right","vessel_location":"proximal","velocity_type":"PSV","flow_direction":"anterograde","source_text":"Proximal ICA PSV 124 cm/s","value":"124"},{"vessel":"ICA","laterality":"Right","vessel_location":"proximal","velocity_type":"EDV","flow_direction":"anterograde","source_text":"PSV 124 cm/s, EDV 42 cm/s","value":"42"},{"vessel":"ICA","laterality":"Right","vessel_location":"distal","velocity_type":"PSV","flow_direction":"anterograde","source_text":"Distal ICA PSV 98 cm/s","value":"98"},{"vessel":"ICA","laterality":"Right","vessel_location":"distal","velocity_type":"EDV","flow_direction":"anterograde","source_text":"PSV 98 cm/s, EDV 31 cm/s","value":"31"},{"vessel":"ECA","laterality":"Right","vessel_location":"InsufficientInformation","velocity_type":"PSV","flow_direction":"anterograde","source_text":"ECA: PSV 96 cm/s","value":"96"},{"vessel":"CCA","laterality":"Left","vessel_location":"InsufficientInformation","velocity_type":"PSV","flow_direction":"anterograde","source_text":"CCA: PSV 78 cm/s","value":"78"},{"vessel":"CCA","laterality":"Left","vessel_location":"InsufficientInformation","velocity_type":"EDV","flow_direction":"anterograde","source_text":"PSV 78 cm/s, EDV 22 cm/s","value":"22"},{"vessel":"ICA","laterality":"Left","vessel_location":"InsufficientInformation","velocity_type":"PSV","flow_direction":"anterograde","source_text":"ICA: PSV 310 cm/s","value":"310"},{"vessel":"ICA","laterality":"Left","vessel_location":"InsufficientInformation","velocity_type":"EDV","flow_direction":"anterograde","source_text":"PSV 310 cm/s, EDV 115 cm/s","value":"115"},{"vessel":"ECA","laterality":"Left","vessel_location":"InsufficientInformation","velocity_type":"PSV","flow_direction":"retrograde","source_text":"ECA: PSV 110 cm/s with retrograde flow","value":"110"}],"stenosis":[{"vessel":"ICA","laterality":"Right","source_text":"Stenosis 40-59%","percentage":"40-59%"},{"vessel":"ICA","laterality":"Left","source_text":"Stenosis 70-79%","percentage":"70-79%"}]}
			""";

	// -------------------------------------------------------------------------
	// Main
	// -------------------------------------------------------------------------

	public static void main(String[] args) throws Exception {
		ObjectMapper mapper = new ObjectMapper();

		Path outputFile = getOutputFile();

		// --- Set up synchronized writer (append mode) ---
		PrintWriter resultWriter = new PrintWriter(new BufferedWriter(
				new OutputStreamWriter(new FileOutputStream(outputFile.toFile(), true), StandardCharsets.UTF_8)));

		Path logFile = getLogFile(outputFile);

		// --- Set up synchronized logger (append mode) ---
		PrintWriter logger = new PrintWriter(new BufferedWriter(
				new OutputStreamWriter(new FileOutputStream(logFile.toFile(), true), StandardCharsets.UTF_8)));

		Object writeLock = new Object();

		// --- Set up the three tab-delimited result files (append mode) ---
		// <output>_ICA_CCA_ratio.tsv, <output>_flow_velocities.tsv, <output>_stenosis.tsv
		VascularTsvWriter tsvWriter = new VascularTsvWriter(outputFile);
		tsvWriter.getFiles().forEach(f -> System.out.println("Tab-delimited results: " + f.toAbsolutePath()));

		// Shared progress counter: incremented once per document that has finished
		// processing, whether it succeeded or failed, so progress and the ETA stay
		// accurate even when some documents fail. AtomicInteger keeps it correct
		// with THREADS > 1.
		AtomicInteger documentNumber = new AtomicInteger(0);

		// Documents that failed (guarded by writeLock). A failure on one document
		// no longer aborts the run; failed paths are written to a re-run list at
		// the end.
		List<Path> failedDocuments = new ArrayList<>();

		// --- Build extractor and run ---
		LlamaVascularExtractor extractor = new LlamaVascularExtractor("http://localhost:8080", "local-model",
				SYSTEM_PROMPT);

		try {
			List<Path> documentPaths = getFilePaths(CORPUS_DIRECTORY_PATH, false);
			int totalDocs = documentPaths.size();

			long beginTime = System.currentTimeMillis();
			System.out.println("Begin processing: " + beginTime + " (" + totalDocs + " documents, THREADS=" + THREADS + ")");

			extractor.analyzeDocumentsParallel(documentPaths, THREADS, result -> {
				String json;
				try {
					json = mapper.writerWithDefaultPrettyPrinter().writeValueAsString(result);
				} catch (Exception e) {
					// Counts as a failure so the document appears in the re-run list.
					throw new RuntimeException("Failed to serialize result for: "
							+ result.documentPathString() + " — " + e.getMessage(), e);
				}
				synchronized (writeLock) {
					int current = documentNumber.incrementAndGet();

					resultWriter.println(json);
					resultWriter.println(); // blank line between records
					resultWriter.flush();

					// Tab-delimited files. A failure here must not mark the document
					// as failed (its JSON is already written), so log and continue.
					try {
						tsvWriter.write(result);
					} catch (Exception tsvException) {
						String msg = "Failed to write tab-delimited rows for "
								+ result.documentPathString() + ": " + tsvException.getMessage();
						System.err.println(msg);
						logger.println(msg);
						logger.flush();
					}

					StringBuilder sb = new StringBuilder();
					sb.append("Result written for document ").append(current).append(" of ")
							.append(totalDocs).append(" documents ").append("\n");
					sb.append("\tFileName: ").append(Paths.get(result.documentPathString()).getFileName()).append("\n");
					double remainingTimeEstimate = estimateRemainingProcessingTime(current, totalDocs, beginTime);
					sb.append(String.format("Estimated remaining processing time: %.2f min%n", remainingTimeEstimate));

					System.out.println(sb.toString());
					logger.println(sb.toString());
					logger.flush();
				}
			}, (path, e) -> {
				synchronized (writeLock) {
					int current = documentNumber.incrementAndGet();
					failedDocuments.add(path);

					StringBuilder sb = new StringBuilder();
					sb.append("FAILED document ").append(current).append(" of ").append(totalDocs)
							.append(" documents ").append("\n");
					sb.append("\tFileName: ").append(path.getFileName()).append("\n");
					sb.append("\tError: ").append(e.getMessage()).append("\n");
					double remainingTimeEstimate = estimateRemainingProcessingTime(current, totalDocs, beginTime);
					sb.append(String.format("Estimated remaining processing time: %.2f min%n", remainingTimeEstimate));

					System.err.println(sb.toString());
					logger.println(sb.toString());
					logger.flush();
				}
			});

			long endTime = System.currentTimeMillis();

			System.out.println("End processing: " + endTime);
			System.out.printf("Total processing time: %.2f min%n", (endTime - beginTime) / 60000.0);

			synchronized (writeLock) {
				int failures = failedDocuments.size();
				String summary = "Succeeded: " + (totalDocs - failures) + "  Failed: " + failures;
				System.out.println(summary);
				logger.println(summary);

				if (failures > 0) {
					Path failedListFile = getFailedListFile(outputFile);
					Files.write(failedListFile,
							failedDocuments.stream().map(Path::toString).collect(Collectors.toList()),
							StandardCharsets.UTF_8);
					String msg = "List of failed documents written to: " + failedListFile.toAbsolutePath();
					System.out.println(msg);
					logger.println(msg);
				}
				logger.flush();
			}

		} catch (IOException e) {
			System.err.println("Unable to load files from " + CORPUS_DIRECTORY_PATH);
			System.err.println(e.getLocalizedMessage());
			e.printStackTrace();
			System.exit(-1);
		} finally {
			resultWriter.close();
			tsvWriter.close();
			logger.close();
		}

	}

	private static Path getLogFile(Path outputPath) {

		String outputFileName = outputPath.toFile().getName();
		int extIndex = outputFileName.indexOf(".");
		String logFileName = extIndex > 0 ? outputFileName.substring(0, extIndex) + ".log" : "logFile.log";

		return outputPath.getParent().resolve(logFileName);
	}

	// e.g. results_261026_101500.json -> results_261026_101500_failed.txt
	private static Path getFailedListFile(Path outputPath) {

		String outputFileName = outputPath.toFile().getName();
		int extIndex = outputFileName.lastIndexOf(".");
		String base = extIndex > 0 ? outputFileName.substring(0, extIndex) : outputFileName;

		return outputPath.getParent().resolve(base + "_failed.txt");
	}

	private static double estimateRemainingProcessingTime(int documentNumber, int totalDocs, long beginTime) {

		long elapsedTime = System.currentTimeMillis() - beginTime;
		int remainingDocuments = totalDocs - documentNumber;

		double processingRate = (1.0 * documentNumber) / elapsedTime;
		double estimatedProcessingTimeLeft = remainingDocuments / processingRate;

		return estimatedProcessingTimeLeft / 60000;
	}

	private static Path getOutputFile() throws IOException {

		try (Scanner scanner = new Scanner(System.in)) {

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
	 * DOCUMENT_EXTENSION and sorted by path. Sorting makes the order (and
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
		Path directory = Paths.get(dirPath);

		if (!Files.exists(directory)) {
			throw new IllegalArgumentException("Path does not exist: " + dirPath);
		}
		if (!Files.isDirectory(directory)) {
			throw new IllegalArgumentException("Path is not a directory: " + dirPath);
		}

		int depth = recursive ? Integer.MAX_VALUE : 1;
		String ext = DOCUMENT_EXTENSION.toLowerCase();

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
