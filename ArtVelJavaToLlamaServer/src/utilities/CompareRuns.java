package utilities;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.PrintStream;
import java.math.BigDecimal;
import java.math.MathContext;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Compares the VALUES extracted by different experiment runs, document by document.
 *
 * <p>It does not score accuracy. It answers: "did these runs extract the same values from each
 * document?" It has no dependencies on the rest of the application (no Jackson, no other
 * packages), so it can be run at any time, also on a computer that has only the output folders.
 *
 * <p>Input: the files each run already wrote into its run folder
 * <pre>
 *   results_flow_velocities.tsv   results_stenosis.tsv   results_ICA_CCA_ratio.tsv
 *   results_documents.tsv         (status and notes per document)
 * </pre>
 * The run folders are found through {@code runs.csv} in the output base directory
 * ({@code output.base-directory} of the machine file). Only the extracted values are compared;
 * offsets, matched text and run_id are ignored. Numbers are compared as numbers (1.20 equals
 * 1.2); text is compared ignoring case and repeated spaces. A document that failed, was too long
 * or is missing in one run counts as different.
 *
 * <h2>Running it (Eclipse)</h2>
 * Run As &gt; Java Application, main class {@code utilities.CompareRuns}, with program arguments such as
 * <pre>
 *   "C:\path\to\output.base-directory" --label gram --set false=gramF --set true=gramT
 * </pre>
 * (use quotes around a path that contains spaces).
 *
 * <h2>Arguments</h2>
 * <pre>
 *   &lt;base&gt;           output.base-directory (the folder with runs.csv); default: current folder
 *   --label REGEX   keep runs whose label matches (case-insensitive, "find" semantics)
 *   --group REGEX   keep runs whose group column matches
 *   --batch TEXT    keep runs whose batch_id contains TEXT
 *   --last N        keep only the N most recent of the selected runs
 *   --runs DIR ...  run folders to include directly (all following arguments up to the next --option)
 *   --set NAME=REGEX  a named set of runs (REGEX is matched against the run label); repeat per set
 *   --out DIR       output folder (default &lt;base&gt;\comparison_yyMMdd_HHmmss)
 * </pre>
 *
 * <h2>Named sets</h2>
 * Within a set, a document is STABLE if every run of the set extracted identical values, and
 * UNSTABLE otherwise (run-to-run noise). Between two sets, a document is a SYSTEMATIC DIFFERENCE only
 * if it is stable in both sets and the two sets' values differ. Documents unstable in either set
 * are counted separately because their differences could be noise.
 *
 * <h2>Output files</h2>
 * <pre>
 *   summary.txt               what is printed on the console
 *   document_by_run.tsv       one row per document: a version letter per run (same letter = same values),
 *                             the number of distinct versions, and the notes per run
 *   pairwise_differences.csv  number of documents that differ for every pair of runs
 *   entry_differences.tsv     for each document with differences: every value and which runs have it
 *   sets_summary.csv          (with --set) stable / unstable counts and between-set differences
 * </pre>
 */
public final class CompareRuns {

	private CompareRuns() {
	}

	// ------------------------------------------------------------------ what is compared
	private static final String DOCUMENTS = "documents";
	private static final String VELOCITY = "velocity";
	private static final String STENOSIS = "stenosis";
	private static final String RATIO = "ratio";

	/** Suffix of the TSV file of each kind (the file is normally named results + suffix). */
	private static final Map<String, String> SUFFIX = new LinkedHashMap<>();
	/** The columns that define the extracted VALUE of an entry (the other columns are ignored). */
	private static final Map<String, List<String>> VALUE_COLUMNS = new LinkedHashMap<>();
	private static final Set<String> NUMERIC_COLUMNS = new HashSet<>(Arrays.asList("value", "percentage", "ratio_value"));

	static {
		SUFFIX.put(VELOCITY, "_flow_velocities.tsv");
		SUFFIX.put(STENOSIS, "_stenosis.tsv");
		SUFFIX.put(RATIO, "_ICA_CCA_ratio.tsv");
		SUFFIX.put(DOCUMENTS, "_documents.tsv");
		VALUE_COLUMNS.put(VELOCITY, Arrays.asList("vessel", "laterality", "vessel_location", "flow_direction",
				"velocity_type", "value", "units"));
		VALUE_COLUMNS.put(STENOSIS, Arrays.asList("vessel", "laterality", "percentage"));
		VALUE_COLUMNS.put(RATIO, Arrays.asList("laterality", "ratio_value"));
	}

	/** Thrown for problems the user can fix (bad arguments, missing folder); printed without a stack trace. */
	private static final class UsageException extends Exception {
		private static final long serialVersionUID = 1L;

		UsageException(String message) {
			super(message);
		}
	}

	// ------------------------------------------------------------------ one run
	private static final class Run {
		final String runId;
		final String label;
		final String group;
		final Path runDir;
		final String repeatIndex;
		String name;
		/** document name -> signature: the sorted extracted entries, or a one-element status marker "&lt;&lt;FAILED&gt;&gt;". */
		final Map<String, List<String>> signatures = new LinkedHashMap<>();
		/** document name -> entry text -> how often it occurs. */
		final Map<String, Map<String, Integer>> entries = new HashMap<>();
		final Map<String, String> status = new HashMap<>();
		final Map<String, String> notes = new HashMap<>();
		int valueCount;
		String problem;

		Run(String runId, String label, String group, Path runDir, String repeatIndex) {
			this.runId = runId;
			this.label = label;
			this.group = group;
			this.runDir = runDir;
			this.repeatIndex = repeatIndex;
			this.name = label.isEmpty() ? runId : label;
		}

		void load() throws IOException {
			if (!Files.isDirectory(runDir)) {
				problem = "run folder not found: " + runDir;
				return;
			}
			Path documentsFile = findFile(runDir, DOCUMENTS);
			if (documentsFile == null) {
				problem = "no *_documents.tsv in " + runDir;
				return;
			}
			// a document processed twice (too long in pass 1, ok in pass 2) -> the last row wins
			for (Map<String, String> row : readTsv(documentsFile)) {
				String doc = row.getOrDefault("DocumentName", "");
				status.put(doc, row.getOrDefault("status", ""));
				notes.put(doc, row.getOrDefault("notes", ""));
			}
			Map<String, Map<String, Integer>> perDocument = new LinkedHashMap<>();
			for (String doc : status.keySet()) {
				perDocument.put(doc, new TreeMap<>());
			}
			for (Map.Entry<String, List<String>> kind : VALUE_COLUMNS.entrySet()) {
				Path file = findFile(runDir, kind.getKey());
				if (file == null) {
					continue;
				}
				for (Map<String, String> row : readTsv(file)) {
					String doc = row.getOrDefault("DocumentName", "");
					StringBuilder entry = new StringBuilder(kind.getKey()).append(": ");
					boolean first = true;
					for (String column : kind.getValue()) {
						if (!first) {
							entry.append(" | ");
						}
						first = false;
						entry.append(column).append('=').append(normalize(column, row.get(column)));
					}
					perDocument.computeIfAbsent(doc, k -> new TreeMap<>()).merge(entry.toString(), 1, Integer::sum);
					valueCount++;
				}
			}
			for (Map.Entry<String, Map<String, Integer>> e : perDocument.entrySet()) {
				String doc = e.getKey();
				entries.put(doc, e.getValue());
				String st = status.getOrDefault(doc, "SUCCESS");
				List<String> signature = new ArrayList<>();
				if (st.startsWith("SUCCESS")) {
					for (Map.Entry<String, Integer> c : e.getValue().entrySet()) {
						for (int i = 0; i < c.getValue(); i++) {
							signature.add(c.getKey());
						}
					}
					Collections.sort(signature);
				} else {
					signature.add("<<" + (st.isEmpty() ? "MISSING" : st) + ">>");
				}
				signatures.put(doc, signature);
			}
		}

		List<String> signature(String doc) {
			List<String> s = signatures.get(doc);
			return s != null ? s : Collections.singletonList("<<MISSING>>");
		}

		int count(String doc, String entry) {
			Map<String, Integer> m = entries.get(doc);
			if (m == null) {
				return 0;
			}
			Integer n = m.get(entry);
			return n == null ? 0 : n;
		}
	}

	// ------------------------------------------------------------------ command line
	private static final class Options {
		Path base = Paths.get(".");
		Pattern label;
		Pattern group;
		String batch;
		int last;
		final List<Path> runs = new ArrayList<>();
		final Map<String, Pattern> sets = new LinkedHashMap<>();
		Path out;
	}

	private static Options parseArguments(String[] args) throws UsageException {
		Options o = new Options();
		boolean baseGiven = false;
		for (int i = 0; i < args.length; i++) {
			String a = args[i];
			switch (a) {
			case "--label":
				o.label = regex(value(args, ++i, a));
				break;
			case "--group":
				o.group = regex(value(args, ++i, a));
				break;
			case "--batch":
				o.batch = value(args, ++i, a);
				break;
			case "--last":
				try {
					o.last = Integer.parseInt(value(args, ++i, a).trim());
				} catch (NumberFormatException e) {
					throw new UsageException("--last needs a whole number");
				}
				break;
			case "--out":
				o.out = Paths.get(value(args, ++i, a));
				break;
			case "--set": {
				String s = value(args, ++i, a);
				int eq = s.indexOf('=');
				if (eq < 1) {
					throw new UsageException("--set needs NAME=REGEX, got: " + s);
				}
				o.sets.put(s.substring(0, eq).trim(), regex(s.substring(eq + 1).trim()));
				break;
			}
			case "--runs":
				while (i + 1 < args.length && !args[i + 1].startsWith("--")) {
					o.runs.add(Paths.get(config.TrunkLocation.expand(args[++i])));
				}
				break;
			default:
				if (a.startsWith("--")) {
					throw new UsageException("Unknown option " + a);
				}
				if (baseGiven) {
					throw new UsageException("More than one base folder given (" + a + "); use --runs for run folders.");
				}
				o.base = Paths.get(config.TrunkLocation.expand(a));
				baseGiven = true;
			}
		}
		return o;
	}

	private static String value(String[] args, int i, String option) throws UsageException {
		if (i >= args.length) {
			throw new UsageException(option + " needs a value");
		}
		return config.TrunkLocation.expand(args[i]);
	}

	private static Pattern regex(String text) throws UsageException {
		try {
			return Pattern.compile(text, Pattern.CASE_INSENSITIVE);
		} catch (RuntimeException e) {
			throw new UsageException("Not a valid regular expression: " + text);
		}
	}

	// ------------------------------------------------------------------ entry point
	public static void main(String[] args) {
		args = config.TrunkLocation.consumeArguments(args); // --trunk=PATH stores the LLM_Trunk location
		int code;
		try {
			code = run(args, System.out);
		} catch (UsageException e) {
			System.err.println(e.getMessage());
			code = 1;
		} catch (IOException e) {
			System.err.println("I/O problem: " + e);
			code = 1;
		}
		if (code != 0) {
			System.exit(code);
		}
	}

	private static int run(String[] args, PrintStream console) throws UsageException, IOException {
		Options o = parseArguments(args);
		List<Run> runs = collectRuns(o, console);
		if (runs.size() < 2) {
			throw new UsageException("Fewer than 2 runs selected (" + runs.size()
					+ "). Check --label / --group or the base folder.");
		}
		setDisplayNames(runs);
		List<Run> usable = new ArrayList<>();
		for (Run r : runs) {
			r.load();
			if (r.problem != null) {
				console.println("  WARNING: " + r.name + ": " + r.problem + " -- run left out");
			} else {
				usable.add(r);
			}
		}
		runs = usable;
		if (runs.size() < 2) {
			throw new UsageException("Fewer than 2 usable runs.");
		}

		Path outDir = o.out != null ? o.out : (Files.isDirectory(o.base) ? o.base : Paths.get("."))
				.resolve("comparison_" + LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyMMdd_HHmmss")));
		Files.createDirectories(outDir);

		Report report = new Report(console);
		compare(runs, o, outDir, report);
		report.say("");
		report.say("Files written to: " + outDir.toAbsolutePath());
		Files.write(outDir.resolve("summary.txt"), (String.join("\n", report.lines) + "\n").getBytes(StandardCharsets.UTF_8));
		return 0;
	}

	/** Prints to the console and keeps every line for summary.txt. */
	private static final class Report {
		final PrintStream console;
		final List<String> lines = new ArrayList<>();

		Report(PrintStream console) {
			this.console = console;
		}

		void say(String text) {
			console.println(text);
			lines.add(text);
		}
	}

	// ------------------------------------------------------------------ choosing the runs
	private static List<Run> collectRuns(Options o, PrintStream console) throws UsageException, IOException {
		List<Run> runs = new ArrayList<>();
		Set<Path> seen = new HashSet<>();
		Path runsCsv = o.base.resolve("runs.csv");
		if (Files.isRegularFile(runsCsv)) {
			for (Map<String, String> row : readCsv(runsCsv)) {
				String runId = row.getOrDefault("run_id", "");
				String label = row.getOrDefault("label", "");
				String group = row.getOrDefault("group", "");
				if (o.label != null && !o.label.matcher(label).find()) {
					continue;
				}
				if (o.group != null && !o.group.matcher(group).find()) {
					continue;
				}
				if (o.batch != null && !row.getOrDefault("batch_id", "").contains(o.batch)) {
					continue;
				}
				String status = row.getOrDefault("status", "");
				if (!status.isEmpty() && !status.equalsIgnoreCase("COMPLETED")) {
					console.println("  skipping " + runId + " (status " + status + ")");
					continue;
				}
				String dir = row.getOrDefault("run_dir", "");
				Path runDir = dir.isEmpty() ? o.base.resolve(runId) : Paths.get(dir);
				if (!Files.isDirectory(runDir)) {
					Path alternative = o.base.resolve(runId); // run folders moved together with the base folder
					if (Files.isDirectory(alternative)) {
						runDir = alternative;
					}
				}
				runs.add(new Run(runId, label, group, runDir, row.getOrDefault("repeat_index", "")));
				seen.add(runDir.toAbsolutePath().normalize());
			}
		} else if (o.runs.isEmpty()) {
			throw new UsageException("No runs.csv in " + o.base.toAbsolutePath()
					+ " (give the output.base-directory, or use --runs).");
		}
		for (Path dir : o.runs) {
			if (seen.contains(dir.toAbsolutePath().normalize())) {
				continue;
			}
			String name = dir.toAbsolutePath().normalize().getFileName().toString();
			runs.add(new Run(name, name, "", dir, ""));
		}
		if (o.last > 0 && runs.size() > o.last) {
			runs = new ArrayList<>(runs.subList(runs.size() - o.last, runs.size()));
		}
		return runs;
	}

	/** Short, unique column names: the label, or label.repeat when the label repeats, or the run id. */
	private static void setDisplayNames(List<Run> runs) {
		Map<String, Integer> labelCounts = new HashMap<>();
		for (Run r : runs) {
			labelCounts.merge(r.label, 1, Integer::sum);
		}
		for (Run r : runs) {
			if (r.label.isEmpty()) {
				r.name = r.runId;
			} else if (labelCounts.get(r.label) == 1) {
				r.name = r.label;
			} else {
				r.name = r.label + "." + (r.repeatIndex.isEmpty()
						? r.runId.substring(Math.max(0, r.runId.length() - 6)) : r.repeatIndex);
			}
		}
		Map<String, Integer> nameCounts = new HashMap<>();
		for (Run r : runs) {
			nameCounts.merge(r.name, 1, Integer::sum);
		}
		for (Run r : runs) {
			if (nameCounts.get(r.name) > 1) {
				r.name = r.runId;
			}
		}
	}

	// ------------------------------------------------------------------ the comparison
	private static void compare(List<Run> runs, Options o, Path outDir, Report report) throws IOException {
		TreeSet<String> docSet = new TreeSet<>();
		for (Run r : runs) {
			docSet.addAll(r.signatures.keySet());
		}
		List<String> docs = new ArrayList<>(docSet);
		int nameWidth = 3;
		for (Run r : runs) {
			nameWidth = Math.max(nameWidth, r.name.length());
		}

		report.say("Compared " + runs.size() + " runs on " + docs.size() + " documents");
		report.say("");
		report.say(String.format("%-" + nameWidth + "s  %6s  %6s  %s", "run", "docs", "values", "documents not SUCCESS"));
		for (Run r : runs) {
			long notOk = r.status.values().stream().filter(s -> !s.startsWith("SUCCESS")).count();
			report.say(String.format("%-" + nameWidth + "s  %6d  %6d  %s", r.name, r.signatures.size(), r.valueCount,
					notOk > 0 ? Long.toString(notOk) : "-"));
		}
		TreeMap<Integer, Integer> coverage = new TreeMap<>();
		for (String d : docs) {
			int n = 0;
			for (Run r : runs) {
				if (r.signatures.containsKey(d)) {
					n++;
				}
			}
			coverage.merge(n, 1, Integer::sum);
		}
		if (coverage.size() > 1 || coverage.firstKey() != runs.size()) {
			report.say("");
			report.say("WARNING: not every document is in every run (documents per number of runs holding them: "
					+ coverage + "). Missing documents count as a difference.");
		}

		// ---- per document: one version letter per run
		Map<String, List<String>> versions = new HashMap<>();
		Map<String, Integer> versionCount = new HashMap<>();
		for (String d : docs) {
			List<List<String>> distinct = new ArrayList<>();
			List<Integer> index = new ArrayList<>();
			for (Run r : runs) {
				List<String> s = r.signature(d);
				int k = distinct.indexOf(s);
				if (k < 0) {
					distinct.add(s);
					k = distinct.size() - 1;
				}
				index.add(k);
			}
			List<String> letters = new ArrayList<>();
			for (int k : index) {
				letters.add(letters(k));
			}
			versions.put(d, letters);
			versionCount.put(d, distinct.size());
		}
		List<String> differing = new ArrayList<>();
		for (String d : docs) {
			if (versionCount.get(d) > 1) {
				differing.add(d);
			}
		}
		report.say("");
		report.say("Documents with identical values in ALL " + runs.size() + " runs: " + (docs.size() - differing.size())
				+ " of " + docs.size());
		report.say("Documents where the runs differ: " + differing.size());
		if (!differing.isEmpty()) {
			report.say("  (" + preview(differing, 40) + ")");
		}

		List<List<String>> rows = new ArrayList<>();
		List<String> header = new ArrayList<>(Arrays.asList("DocumentName", "distinct_versions", "all_same"));
		for (Run r : runs) {
			header.add(r.name);
		}
		for (Run r : runs) {
			header.add("notes: " + r.name);
		}
		rows.add(header);
		for (String d : docs) {
			List<String> row = new ArrayList<>(Arrays.asList(d, Integer.toString(versionCount.get(d)),
					versionCount.get(d) == 1 ? "yes" : "NO"));
			row.addAll(versions.get(d));
			for (Run r : runs) {
				row.add(r.notes.getOrDefault(d, ""));
			}
			rows.add(row);
		}
		writeTable(outDir.resolve("document_by_run.tsv"), rows, '\t');

		// ---- pairwise
		int n = runs.size();
		int[][] pair = new int[n][n];
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				int count = 0;
				for (String d : docs) {
					if (!runs.get(i).signature(d).equals(runs.get(j).signature(d))) {
						count++;
					}
				}
				pair[i][j] = count;
			}
		}
		rows = new ArrayList<>();
		List<String> pairHeader = new ArrayList<>();
		pairHeader.add("");
		for (Run r : runs) {
			pairHeader.add(r.name);
		}
		rows.add(pairHeader);
		for (int i = 0; i < n; i++) {
			List<String> row = new ArrayList<>();
			row.add(runs.get(i).name);
			for (int j = 0; j < n; j++) {
				row.add(Integer.toString(pair[i][j]));
			}
			rows.add(row);
		}
		writeTable(outDir.resolve("pairwise_differences.csv"), rows, ',');
		report.say("");
		report.say("Documents that differ, for each pair of runs (also in pairwise_differences.csv):");
		StringBuilder head = new StringBuilder(String.format("%-" + nameWidth + "s  ", ""));
		for (int j = 0; j < n; j++) {
			head.append(String.format("%3d  ", j + 1));
		}
		report.say(head.toString().replaceAll("\\s+$", ""));
		for (int i = 0; i < n; i++) {
			StringBuilder line = new StringBuilder(String.format("%-" + nameWidth + "s  ", runs.get(i).name));
			for (int j = 0; j < n; j++) {
				line.append(String.format("%3d  ", pair[i][j]));
			}
			report.say(line.toString().replaceAll("\\s+$", "") + " <- " + (i + 1));
		}

		// ---- entry level detail
		rows = new ArrayList<>();
		rows.add(Arrays.asList("DocumentName", "entry", "runs_with_entry", "of_runs", "runs_that_have_it",
				"runs_that_lack_it"));
		for (String d : differing) {
			Set<String> keys = new java.util.LinkedHashSet<>();
			for (Run r : runs) {
				Map<String, Integer> m = r.entries.get(d);
				if (m != null) {
					keys.addAll(m.keySet());
				}
			}
			for (String k : keys) {
				Set<Integer> counts = new HashSet<>();
				List<String> have = new ArrayList<>();
				List<String> lack = new ArrayList<>();
				for (Run r : runs) {
					int c = r.count(d, k);
					counts.add(c);
					if (c > 0) {
						have.add(r.name + (c == 1 ? "" : " (x" + c + ")"));
					} else {
						lack.add(r.name);
					}
				}
				if (counts.size() == 1) {
					continue; // the same number of times in every run: not a difference
				}
				rows.add(Arrays.asList(d, k, Integer.toString(have.size()), Integer.toString(n),
						String.join("; ", have), String.join("; ", lack)));
			}
			// a document that was not extracted in some run (FAILED, TOO_LONG ...) is a difference too
			for (Run r : runs) {
				String first = r.signature(d).get(0);
				if (first.startsWith("<<")) {
					rows.add(Arrays.asList(d, "STATUS " + first, "1", Integer.toString(n), r.name, ""));
				}
			}
		}
		writeTable(outDir.resolve("entry_differences.tsv"), rows, '\t');

		// ---- do retries / repairs go with the differences?
		List<String> flagged = new ArrayList<>();
		for (String d : docs) {
			for (Run r : runs) {
				String note = r.notes.getOrDefault(d, "");
				if (note.contains("JSON_REPAIRED") || note.contains("RETRIED")) {
					flagged.add(d);
					break;
				}
			}
		}
		if (!flagged.isEmpty()) {
			int both = 0;
			for (String d : differing) {
				if (flagged.contains(d)) {
					both++;
				}
			}
			report.say("");
			report.say("Documents with a retry or JSON repair in at least one run: " + flagged.size() + "; of those, "
					+ both + " differ between runs (vs " + (differing.size() - both) + " of the other "
					+ (docs.size() - flagged.size()) + ").");
		}

		if (!o.sets.isEmpty()) {
			compareSets(runs, docs, o, outDir, report);
		}
	}

	private static void compareSets(List<Run> runs, List<String> docs, Options o, Path outDir, Report report)
			throws IOException {
		Map<String, List<Run>> members = new LinkedHashMap<>();
		for (String name : o.sets.keySet()) {
			members.put(name, new ArrayList<>());
		}
		for (Run r : runs) {
			for (Map.Entry<String, Pattern> s : o.sets.entrySet()) {
				if (s.getValue().matcher(r.label).find()) {
					members.get(s.getKey()).add(r);
					break;
				}
			}
		}
		report.say("");
		report.say("SETS");
		Map<String, Set<String>> stable = new LinkedHashMap<>();
		List<List<String>> setRows = new ArrayList<>();
		setRows.add(Arrays.asList("set", "runs", "identical_in_all_runs_of_set", "unstable_documents"));
		for (Map.Entry<String, List<Run>> m : members.entrySet()) {
			List<Run> rs = m.getValue();
			if (rs.isEmpty()) {
				report.say("  set '" + m.getKey() + "': no runs match");
				continue;
			}
			Set<String> stableDocs = new TreeSet<>();
			List<String> unstable = new ArrayList<>();
			for (String d : docs) {
				Set<List<String>> distinct = new HashSet<>();
				for (Run r : rs) {
					distinct.add(r.signature(d));
				}
				if (distinct.size() == 1) {
					stableDocs.add(d);
				} else {
					unstable.add(d);
				}
			}
			stable.put(m.getKey(), stableDocs);
			report.say("  set '" + m.getKey() + "': " + rs.size() + " runs ("
					+ rs.stream().map(r -> r.name).collect(Collectors.joining(", ")) + ")");
			if (rs.size() > 1) {
				report.say("     identical in every run of the set: " + stableDocs.size() + " documents; unstable (runs differ): "
						+ unstable.size() + (unstable.isEmpty() ? "" : "  -> " + preview(unstable, 30)));
			}
			setRows.add(Arrays.asList(m.getKey(), Integer.toString(rs.size()), Integer.toString(stableDocs.size()),
					Integer.toString(unstable.size())));
		}
		List<String> names = new ArrayList<>(stable.keySet());
		setRows.add(new ArrayList<>());
		setRows.add(Arrays.asList("set_a", "set_b", "docs_stable_in_both", "systematic_differences",
				"docs_unstable_in_either"));
		for (int i = 0; i < names.size(); i++) {
			for (int j = i + 1; j < names.size(); j++) {
				String a = names.get(i);
				String b = names.get(j);
				List<String> bothStable = new ArrayList<>();
				List<String> systematic = new ArrayList<>();
				for (String d : docs) {
					if (stable.get(a).contains(d) && stable.get(b).contains(d)) {
						bothStable.add(d);
						if (!members.get(a).get(0).signature(d).equals(members.get(b).get(0).signature(d))) {
							systematic.add(d);
						}
					}
				}
				int unstableEither = docs.size() - bothStable.size();
				report.say("");
				report.say("  '" + a + "' vs '" + b + "':");
				report.say("     documents stable in both sets: " + bothStable.size());
				report.say("     of those, values differ between the sets (systematic difference): " + systematic.size()
						+ (systematic.isEmpty() ? "" : "  -> " + preview(systematic, 30)));
				report.say("     documents unstable in at least one set (cannot be attributed to the setting): "
						+ unstableEither);
				setRows.add(Arrays.asList(a, b, Integer.toString(bothStable.size()), Integer.toString(systematic.size()),
						Integer.toString(unstableEither)));
			}
		}
		writeTable(outDir.resolve("sets_summary.csv"), setRows, ',');
	}

	// ------------------------------------------------------------------ small helpers
	/** 0 -&gt; A, 1 -&gt; B, ... 25 -&gt; Z, 26 -&gt; AA. */
	private static String letters(int k) {
		StringBuilder s = new StringBuilder();
		int x = k;
		do {
			s.insert(0, (char) ('A' + x % 26));
			x = x / 26 - 1;
		} while (x >= 0);
		return s.toString();
	}

	private static String preview(List<String> items, int max) {
		String text = items.stream().limit(max).collect(Collectors.joining(", "));
		return items.size() > max ? text + " ..." : text;
	}

	/** Numbers are compared as numbers (to 6 significant digits); text ignoring case and repeated spaces. */
	private static String normalize(String column, String text) {
		String t = text == null ? "" : text.trim();
		if (NUMERIC_COLUMNS.contains(column)) {
			try {
				BigDecimal d = new BigDecimal(t).round(new MathContext(6));
				return d.signum() == 0 ? "0" : d.stripTrailingZeros().toPlainString();
			} catch (NumberFormatException e) {
				// not a number (for example NoInformation): fall through to text
			}
		}
		return t.replaceAll("\\s+", " ").toLowerCase(Locale.ROOT);
	}

	/** The run folder's TSV of the given kind: results + suffix, or any file ending with the suffix. */
	private static Path findFile(Path runDir, String kind) throws IOException {
		String suffix = SUFFIX.get(kind);
		Path preferred = runDir.resolve("results" + suffix);
		if (Files.isRegularFile(preferred)) {
			return preferred;
		}
		try (Stream<Path> list = Files.list(runDir)) {
			return list.filter(p -> p.getFileName().toString().endsWith(suffix)).sorted().findFirst().orElse(null);
		}
	}

	// ------------------------------------------------------------------ reading and writing tables
	/** Reads a tab-delimited file written by VascularTsvWriter (no quoting; tabs inside fields are escaped). */
	private static List<Map<String, String>> readTsv(Path file) throws IOException {
		List<Map<String, String>> rows = new ArrayList<>();
		try (BufferedReader in = Files.newBufferedReader(file, StandardCharsets.UTF_8)) {
			String line = in.readLine();
			if (line == null) {
				return rows;
			}
			String[] header = stripBom(line).split("\t", -1);
			while ((line = in.readLine()) != null) {
				if (line.isEmpty()) {
					continue;
				}
				String[] cells = line.split("\t", -1);
				Map<String, String> row = new HashMap<>();
				for (int i = 0; i < header.length; i++) {
					row.put(header[i], i < cells.length ? cells[i] : "");
				}
				rows.add(row);
			}
		}
		return rows;
	}

	/** Reads a comma-separated file with quoted fields ("" for a quote; quoted fields may contain line breaks). */
	private static List<Map<String, String>> readCsv(Path file) throws IOException {
		String text = stripBom(new String(Files.readAllBytes(file), StandardCharsets.UTF_8));
		List<List<String>> table = new ArrayList<>();
		List<String> row = new ArrayList<>();
		StringBuilder cell = new StringBuilder();
		boolean quoted = false;
		for (int i = 0; i < text.length(); i++) {
			char c = text.charAt(i);
			if (quoted) {
				if (c == '"') {
					if (i + 1 < text.length() && text.charAt(i + 1) == '"') {
						cell.append('"');
						i++;
					} else {
						quoted = false;
					}
				} else {
					cell.append(c);
				}
			} else if (c == '"') {
				quoted = true;
			} else if (c == ',') {
				row.add(cell.toString());
				cell.setLength(0);
			} else if (c == '\n' || c == '\r') {
				if (c == '\r' && i + 1 < text.length() && text.charAt(i + 1) == '\n') {
					i++;
				}
				row.add(cell.toString());
				cell.setLength(0);
				table.add(row);
				row = new ArrayList<>();
			} else {
				cell.append(c);
			}
		}
		if (cell.length() > 0 || !row.isEmpty()) {
			row.add(cell.toString());
			table.add(row);
		}
		List<Map<String, String>> result = new ArrayList<>();
		if (table.isEmpty()) {
			return result;
		}
		List<String> header = table.get(0);
		for (int r = 1; r < table.size(); r++) {
			List<String> cells = table.get(r);
			if (cells.size() == 1 && cells.get(0).isEmpty()) {
				continue; // blank line
			}
			Map<String, String> map = new HashMap<>();
			for (int i = 0; i < header.size(); i++) {
				map.put(header.get(i), i < cells.size() ? cells.get(i) : "");
			}
			result.add(map);
		}
		return result;
	}

	private static String stripBom(String s) {
		return s.startsWith("\uFEFF") ? s.substring(1) : s;
	}

	/** Writes a table with a UTF-8 byte order mark (so Excel reads it correctly). */
	private static void writeTable(Path file, List<List<String>> rows, char separator) throws IOException {
		StringBuilder out = new StringBuilder("\uFEFF");
		for (List<String> row : rows) {
			for (int i = 0; i < row.size(); i++) {
				if (i > 0) {
					out.append(separator);
				}
				out.append(cell(row.get(i), separator));
			}
			out.append('\n');
		}
		Files.write(file, out.toString().getBytes(StandardCharsets.UTF_8));
	}

	private static String cell(String value, char separator) {
		String v = value == null ? "" : value;
		if (separator == '\t') {
			return v.replace('\t', ' ').replace('\r', ' ').replace('\n', ' ');
		}
		if (v.indexOf(',') >= 0 || v.indexOf('"') >= 0 || v.indexOf('\n') >= 0 || v.indexOf('\r') >= 0) {
			return "\"" + v.replace("\"", "\"\"") + "\"";
		}
		return v;
	}
}
