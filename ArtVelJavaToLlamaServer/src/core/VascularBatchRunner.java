package core;

import config.AppSettings;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.time.Instant;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Runs a list of experiment files one after another with no prompts (Step 3 of the experiment plan).
 *
 * <h3>Batch file format</h3>
 * One experiment .properties file per line. Blank lines and lines starting with # are ignored.
 * Relative paths are resolved against the folder containing the batch file. To repeat an
 * experiment, end the line with {@code * N}:
 * <pre>
 * # baseline, three times
 * experiments/baseline.properties * 3
 * experiments/prompt-variant.properties
 * </pre>
 * Every run uses the same machine file (--config / -Dvascular.config / VASCULAR_CONFIG, or the
 * built-in VA defaults) and its own fresh run folder. A run that fails is recorded (ABORTED in
 * runs.csv where a run folder exists) and the batch continues with the next run.
 * A status file {@code batch_<id>_status.csv} is written in the machine file's output.base-directory
 * (or next to the batch file if that is not set).
 */
public final class VascularBatchRunner {

	private static final Pattern REPEAT = Pattern.compile("^(.*\\S)\\s+\\*\\s*(\\d+)$");
	private static final String STATUS_HEADER = "batch_id\tentry\texperiment_file\trepeat\tstatus\trun_id\tseconds\tdetail";

	private VascularBatchRunner() {
	}

	private static final class Entry {
		final Path file;
		final int repeats;
		final int lineNumber;

		Entry(Path file, int repeats, int lineNumber) {
			this.file = file;
			this.repeats = repeats;
			this.lineNumber = lineNumber;
		}
	}

	/** Reads and validates the batch file. Throws IOException listing every problem found. */
	static List<Entry> readBatch(Path batchFile) throws IOException {
		Path base = batchFile.toAbsolutePath().getParent();
		List<String> lines = Files.readAllLines(batchFile, StandardCharsets.UTF_8);
		List<Entry> entries = new ArrayList<>();
		List<String> problems = new ArrayList<>();
		for (int i = 0; i < lines.size(); i++) {
			String line = lines.get(i);
			if (i == 0 && line.startsWith("\uFEFF")) {
				line = line.substring(1);
			}
			line = line.trim();
			if (line.isEmpty() || line.startsWith("#")) {
				continue;
			}
			int repeats = 1;
			Matcher m = REPEAT.matcher(line);
			if (m.matches()) {
				line = m.group(1).trim();
				repeats = Integer.parseInt(m.group(2));
				if (repeats < 1 || repeats > 1000) {
					problems.add("line " + (i + 1) + ": repeat count must be 1 to 1000");
					continue;
				}
			}
			if (line.length() >= 2 && line.startsWith("\"") && line.endsWith("\"")) {
				line = line.substring(1, line.length() - 1);
			}
			if (line.equals("~") || line.startsWith("~/") || line.startsWith("~\\")) {
				line = System.getProperty("user.home") + line.substring(1);
			}
			Path f = Paths.get(line);
			if (!f.isAbsolute()) {
				f = base.resolve(f);
			}
			f = f.normalize();
			if (!Files.isRegularFile(f)) {
				problems.add("line " + (i + 1) + ": file not found: " + f);
				continue;
			}
			entries.add(new Entry(f, repeats, i + 1));
		}
		if (entries.isEmpty() && problems.isEmpty()) {
			problems.add("the batch file lists no experiment files");
		}
		if (!problems.isEmpty()) {
			throw new IOException("Batch file " + batchFile + ":\n  " + String.join("\n  ", problems));
		}
		return entries;
	}

	/**
	 * Runs every entry of the batch file. Nothing is run if the batch file itself has problems.
	 * Returns the number of runs that did not complete.
	 */
	public static int runBatch(Path batchFile) throws IOException {
		List<Entry> entries = readBatch(batchFile);
		Path machine = AppSettings.resolveMachineFile(null); // never prompts

		String batchId = DateTimeFormatter.ofPattern("yyMMdd_HHmmss").withZone(ZoneId.systemDefault())
				.format(Instant.now());
		int totalRuns = 0;
		for (Entry e : entries) {
			totalRuns += e.repeats;
		}

		Path statusDir = batchFile.toAbsolutePath().getParent();
		try {
			Path base = AppSettings.fromFiles(machine, null).outputBaseDirectory();
			if (base != null) {
				Files.createDirectories(base);
				statusDir = base;
			}
		} catch (Exception ex) {
			// Fall back to the batch file's folder; individual runs report their own settings errors.
		}
		Path statusFile = statusDir.resolve("batch_" + batchId + "_status.csv");
		appendStatus(statusFile, STATUS_HEADER);

		System.out.println("==== Batch " + batchId + ": " + entries.size() + " experiment file(s), " + totalRuns
				+ " run(s). Machine settings: " + (machine == null ? "built-in VA defaults" : machine) + " ====");
		System.out.println("Batch status file: " + statusFile.toAbsolutePath());

		int runNumber = 0;
		int notCompleted = 0;
		for (int ei = 0; ei < entries.size(); ei++) {
			Entry e = entries.get(ei);
			for (int rep = 1; rep <= e.repeats; rep++) {
				runNumber++;
				System.out.println();
				System.out.println("==== Batch " + batchId + ": run " + runNumber + " of " + totalRuns + " -- "
						+ e.file.getFileName() + " (repeat " + rep + " of " + e.repeats + ") ====");
				long t0 = System.nanoTime();
				String status;
				String runId = "";
				String detail = "";
				try {
					AppSettings s = AppSettings.fromFiles(machine, e.file);
					MainVascular.RunOutcome out = MainVascular.run(s, null, batchId, rep);
					status = out.status;
					runId = out.runId == null ? "" : out.runId;
					detail = out.summary;
				} catch (Throwable t) { // includes OutOfMemoryError etc.: log and go on
					status = "ABORTED";
					detail = t.getClass().getSimpleName() + ": " + String.valueOf(t.getLocalizedMessage()).replace('\n', ' ');
					System.err.println("Run failed, continuing with the next one: " + detail);
					t.printStackTrace();
				}
				if (!"COMPLETED".equals(status)) {
					notCompleted++;
				}
				double seconds = (System.nanoTime() - t0) / 1e9;
				appendStatus(statusFile, String.join("\t", batchId, Integer.toString(ei + 1),
						e.file.toString(), Integer.toString(rep), status, runId,
						String.format(Locale.ROOT, "%.1f", seconds), detail.replace('\t', ' ')));
				System.out.println("==== Run " + runNumber + " of " + totalRuns + ": " + status
						+ (runId.isEmpty() ? "" : " (" + runId + ")") + " ====");
				if (runNumber < totalRuns) {
					pauseBetweenRuns();
				}
			}
		}
		System.out.println();
		System.out.println("==== Batch " + batchId + " finished: " + (totalRuns - notCompleted) + " of " + totalRuns
				+ " run(s) completed ====");
		return notCompleted;
	}

	/** Gives the operating system a moment to release the server's port and memory. */
	private static void pauseBetweenRuns() {
		try {
			Thread.sleep(5000);
		} catch (InterruptedException ie) {
			Thread.currentThread().interrupt();
		}
	}

	private static void appendStatus(Path file, String line) {
		try {
			Files.write(file, (line + System.lineSeparator()).getBytes(StandardCharsets.UTF_8),
					StandardOpenOption.CREATE, StandardOpenOption.APPEND);
		} catch (IOException ex) {
			System.err.println("Could not update the batch status file: " + ex.getLocalizedMessage());
		}
	}
}
