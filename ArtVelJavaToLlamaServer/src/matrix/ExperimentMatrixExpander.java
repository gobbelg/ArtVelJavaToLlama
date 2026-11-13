package matrix;

import config.AppSettings;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Deque;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Expands a matrix definition file into experiment files and a batch file.
 *
 * <p>Run: {@code matrix.ExperimentMatrixExpander <matrix-file> [--overwrite]}
 *
 * <h3>Matrix file</h3>
 * Plain text. Blank lines and lines starting with # are ignored. Relative paths are relative to the
 * folder of the file that contains them.
 * <pre>
 * # ---- global settings (before the first [group]) ----
 * base.experiment  = path of an experiment .properties file (required): everything not varied comes from it
 * machine.config   = path of the machine .properties file (recommended: needed for "+" keys and for validation)
 * output.directory = folder for the generated files (required)
 * slot.size        = tokens per slot (optional). When a run sets run.default-parallel and not server.ctx-size,
 *                    server.ctx-size = run.default-parallel * slot.size
 * corpus.directory = folder with the reports (optional). Creates document_lists/corpus_all.txt and a
 *                    size-spread subset list corpus_&lt;subset.name&gt;_&lt;subset.size&gt;.txt
 * corpus.extension = .txt          subset.size = 30        subset.name = screen
 * estimate.minutes.per.run = 10    (optional: prints a total-time estimate)
 * include = other-matrix-file      (optional: splices in the [groups] of another file)
 * only.groups = g1, g2             (optional: expand ONLY these groups)
 * skip.groups = g3, g4             (optional: leave these groups out, e.g. stages already run)
 * sampler = path of sample_resources_windows.bat   (optional: the batch starts/stops this resource sampler)
 * sampler.interval = 5           (optional: seconds between samples)
 *
 * # ---- groups ----
 * [s1_parallel]
 * description = free text
 * repeat   = 1                     (each generated experiment runs this many times)
 * shuffle  = true                  (random order within the group; shuffle.seed = 1 makes it repeatable)
 * documents = all | screen | base | path-of-list-file
 * estimate.minutes = 10            (overrides the global estimate for this group)
 * set  server.threads = 4          (fixed for every run in the group)
 * vary run.default-parallel = 1 | 2 | 4 | 8     (one run per value; several "vary" lines = every combination)
 * vary +server.extra-args = f16= | q8=--cache-type-k q8_0 --cache-type-v q8_0
 * </pre>
 * A leading "+" on the key appends to the existing value (the base experiment's, else the machine file's)
 * instead of replacing it. A value may start with {@code alias=}; the alias is used in labels and the design table.
 */
public final class ExperimentMatrixExpander {

	private static final Set<String> GLOBAL_KEYS = new LinkedHashSet<>(java.util.Arrays.asList(
			"base.experiment", "machine.config", "output.directory", "slot.size", "corpus.directory",
			"corpus.extension", "subset.size", "subset.name", "estimate.minutes.per.run", "include",
			"only.groups", "skip.groups", "sampler", "sampler.interval"));
	private static final Set<String> GROUP_KEYS = new LinkedHashSet<>(java.util.Arrays.asList(
			"description", "repeat", "shuffle", "shuffle.seed", "documents", "estimate.minutes"));
	private static final Set<String> PATH_KEYS = new LinkedHashSet<>(java.util.Arrays.asList(
			"prompt.file", "experiment.document-list", "corpus.directory", "server.model",
			"server.executable", "output.base-directory"));
	private static final Pattern ALIAS = Pattern.compile("^([A-Za-z][A-Za-z0-9_\\-]*)=(.*)$");
	private static final Pattern GROUP_NAME = Pattern.compile("^[A-Za-z0-9][A-Za-z0-9_\\-]*$");

	private ExperimentMatrixExpander() {
	}

	// ------------------------------------------------------------------ model

	private static final class Alt {
		final String alias; // may be null
		final String value;

		Alt(String alias, String value) {
			this.alias = alias;
			this.value = value;
		}
	}

	private static final class Vary {
		final String key;
		final boolean append;
		final List<Alt> alts = new ArrayList<>();

		Vary(String key, boolean append) {
			this.key = key;
			this.append = append;
		}
	}

	private static final class Group {
		final String name;
		String description = "";
		int repeat = 1;
		boolean shuffle;
		long seed = 1;
		String documents; // null = default
		Double estimateMinutes;
		final List<String[]> sets = new ArrayList<>(); // {key, value, appendFlag "1"/"0"}
		final List<Vary> varies = new ArrayList<>();
		Path baseDir;

		Group(String name) {
			this.name = name;
		}
	}

	private static final class Run {
		final Group group;
		final String label;
		final Map<String, String> overrides = new LinkedHashMap<>(); // key -> final value
		final Map<String, String> varied = new LinkedHashMap<>(); // key -> shown value (alias or value)
		Path file;
		String documentsShown = "base";

		Run(Group group, String label) {
			this.group = group;
			this.label = label;
		}
	}

	private static final class SrcLine {
		final String text;
		final Path file;
		final int number;

		SrcLine(String text, Path file, int number) {
			this.text = text;
			this.file = file;
			this.number = number;
		}

		String where() {
			return file.getFileName() + " line " + number;
		}
	}

	// ------------------------------------------------------------------ entry point

	public static void main(String[] args) {
		args = config.TrunkLocation.consumeArguments(args); // --trunk=PATH stores the LLM_Trunk location
		if (args.length < 1) {
			System.err.println("Usage: matrix.ExperimentMatrixExpander <matrix-file> [--overwrite] [--trunk=<LLM_Trunk folder>]");
			System.exit(2);
		}
		boolean overwrite = false;
		for (int i = 1; i < args.length; i++) {
			if (args[i].equals("--overwrite")) {
				overwrite = true;
			}
		}
		int code;
		try {
			// Files written by the expander use <LLM_TRUNK> instead of this computer's path, so they
			// stay valid for every user. That needs the location now (asks the user if it is not set).
			config.TrunkLocation.ensure();
			code = expand(Paths.get(config.TrunkLocation.expand(args[0])).toAbsolutePath().normalize(), overwrite);
		} catch (Exception e) {
			System.err.println("Failed: " + e.getLocalizedMessage());
			e.printStackTrace();
			code = 1;
		}
		System.exit(code);
	}

	/** Returns 0 when everything was generated and validated; 1 otherwise. */
	public static int expand(Path matrixFile, boolean overwrite) throws IOException {
		List<String> errors = new ArrayList<>();
		List<String> warnings = new ArrayList<>();

		// ---- read the file (with includes)
		List<SrcLine> lines = new ArrayList<>();
		readLines(matrixFile, new ArrayDeque<Path>(), lines, errors);

		Map<String, String> global = new LinkedHashMap<>();
		Map<String, SrcLine> globalAt = new LinkedHashMap<>();
		List<Group> groups = new ArrayList<>();
		Group cur = null;
		for (SrcLine sl : lines) {
			String t = sl.text.trim();
			if (t.isEmpty() || t.startsWith("#")) {
				continue;
			}
			if (t.startsWith("[")) {
				if (!t.endsWith("]")) {
					errors.add(sl.where() + ": group header must look like [name]");
					continue;
				}
				String name = t.substring(1, t.length() - 1).trim();
				if (!GROUP_NAME.matcher(name).matches()) {
					errors.add(sl.where() + ": group name may use letters, digits, _ and - only: " + name);
					continue;
				}
				for (Group g : groups) {
					if (g.name.equals(name)) {
						errors.add(sl.where() + ": group '" + name + "' is defined twice");
					}
				}
				cur = new Group(name);
				cur.baseDir = sl.file.getParent();
				groups.add(cur);
				continue;
			}
			if (t.equals("@end-include")) { // marker added by readLines after an include
				cur = null;
				continue;
			}
			String first = t.split("\\s+", 2)[0];
			if (first.equals("set") || first.equals("vary")) {
				if (cur == null) {
					errors.add(sl.where() + ": '" + first + "' must come after a [group] header");
					continue;
				}
				String rest = t.substring(first.length()).trim();
				int eq = rest.indexOf('=');
				if (eq < 1) {
					errors.add(sl.where() + ": expected '" + first + " key = value'");
					continue;
				}
				String key = rest.substring(0, eq).trim();
				boolean append = key.startsWith("+");
				if (append) {
					key = key.substring(1).trim();
				}
				String value = rest.substring(eq + 1).trim();
				if (key.isEmpty() || GLOBAL_KEYS.contains(key) || GROUP_KEYS.contains(key)) {
					errors.add(sl.where() + ": not a setting that can be set or varied: '" + key + "'");
					continue;
				}
				if (first.equals("set")) {
					cur.sets.add(new String[] { key, unquote(value), append ? "1" : "0" });
				} else {
					Vary v = new Vary(key, append);
					for (String part : value.split("\\|", -1)) {
						String p = part.trim();
						String alias = null;
						Matcher m = ALIAS.matcher(p);
						if (m.matches()) {
							alias = m.group(1);
							p = m.group(2).trim();
						}
						if (p.isEmpty() && alias == null) {
							errors.add(sl.where() + ": empty alternative in 'vary " + key + "' (give it an alias, e.g. none=)");
							continue;
						}
						v.alts.add(new Alt(alias, unquote(p)));
					}
					if (v.alts.isEmpty()) {
						errors.add(sl.where() + ": 'vary " + key + "' lists no values");
					} else {
						cur.varies.add(v);
					}
				}
				continue;
			}
			int eq = t.indexOf('=');
			if (eq < 1) {
				errors.add(sl.where() + ": expected key = value, found: " + t);
				continue;
			}
			String key = t.substring(0, eq).trim();
			String value = unquote(t.substring(eq + 1).trim());
			if (cur != null && GROUP_KEYS.contains(key)) {
				try {
					switch (key) {
					case "description":
						cur.description = value;
						break;
					case "repeat":
						cur.repeat = Integer.parseInt(value);
						if (cur.repeat < 1 || cur.repeat > 1000) {
							throw new NumberFormatException();
						}
						break;
					case "shuffle":
						cur.shuffle = parseBool(value);
						break;
					case "shuffle.seed":
						cur.seed = Long.parseLong(value);
						break;
					case "documents":
						cur.documents = value;
						break;
					case "estimate.minutes":
						cur.estimateMinutes = Double.parseDouble(value);
						break;
					default:
						break;
					}
				} catch (RuntimeException e) {
					errors.add(sl.where() + ": bad value for " + key + ": " + value);
				}
			} else if (GLOBAL_KEYS.contains(key)) {
				if (cur != null) {
					errors.add(sl.where() + ": global setting '" + key + "' must come before the first [group]");
				} else if (key.equals("include")) {
					// handled in readLines
				} else {
					if (global.containsKey(key)) {
						errors.add(sl.where() + ": '" + key + "' is set twice");
					}
					global.put(key, value);
					globalAt.put(key, sl);
				}
			} else {
				errors.add(sl.where() + ": unknown setting '" + key + "'");
			}
		}

		// ---- global settings
		Path base = pathOf(global, globalAt, "base.experiment", true, errors);
		Path machine = pathOf(global, globalAt, "machine.config", false, errors);
		Path outDir = pathOf(global, globalAt, "output.directory", true, errors);
		Path corpusDir = pathOf(global, globalAt, "corpus.directory", false, errors);
		Path samplerScript = pathOf(global, globalAt, "sampler", false, errors);
		Integer samplerInterval = intOf(global, "sampler.interval", 1, 3600, errors);
		if (samplerScript != null && !Files.isRegularFile(samplerScript)) {
			errors.add("sampler script not found: " + samplerScript);
		} else if (samplerScript != null && samplerScript.getFileName().toString().indexOf(' ') >= 0) {
			errors.add("the sampler script's file name must not contain spaces: " + samplerScript.getFileName());
		}
		Integer slot = intOf(global, "slot.size", 256, 1_000_000, errors);
		Integer subsetSize = intOf(global, "subset.size", 1, 100000, errors);
		String ext = global.getOrDefault("corpus.extension", ".txt");
		String subsetName = global.getOrDefault("subset.name", "screen");
		Double estimateGlobal = null;
		if (global.containsKey("estimate.minutes.per.run")) {
			try {
				estimateGlobal = Double.parseDouble(global.get("estimate.minutes.per.run"));
			} catch (NumberFormatException e) {
				errors.add("estimate.minutes.per.run must be a number");
			}
		}
		if (base != null && !Files.isRegularFile(base)) {
			errors.add("base.experiment not found: " + base);
		}
		if (machine != null && !Files.isRegularFile(machine)) {
			errors.add("machine.config not found: " + machine);
		}
		if (corpusDir != null && !Files.isDirectory(corpusDir)) {
			errors.add("corpus.directory not found: " + corpusDir);
		}
		if (groups.isEmpty()) {
			errors.add("the matrix file defines no [group]");
		}
		Set<String> only = nameList(global.get("only.groups"));
		Set<String> skip = nameList(global.get("skip.groups"));
		Set<String> known = new LinkedHashSet<>();
		for (Group g : groups) {
			known.add(g.name);
		}
		for (String nme : only) {
			if (!known.contains(nme)) {
				errors.add("only.groups names an unknown group: " + nme);
			}
		}
		for (String nme : skip) {
			if (!known.contains(nme)) {
				errors.add("skip.groups names an unknown group: " + nme);
			}
		}
		final List<Group> kept = new ArrayList<>();
		for (Group g : groups) {
			if ((only.isEmpty() || only.contains(g.name)) && !skip.contains(g.name)) {
				kept.add(g);
			}
		}
		if (kept.isEmpty() && errors.isEmpty()) {
			errors.add("only.groups / skip.groups leave no group to expand");
		}
		groups = kept;
		if (!errors.isEmpty()) {
			return fail(errors);
		}

		Map<String, String> baseValues = readKeyValues(base);
		Map<String, String> machineValues = machine == null ? new LinkedHashMap<String, String>() : readKeyValues(machine);
		List<String> baseLines = new ArrayList<>();
		for (String bl : Files.readAllLines(base, StandardCharsets.UTF_8)) {
			baseLines.add(config.TrunkLocation.expand(bl));
		}

		// ---- output folders
		Path expDir = outDir.resolve("experiments");
		Path listDir = outDir.resolve("document_lists");
		String matrixName = stripExtension(matrixFile.getFileName().toString());
		Path batchFile = outDir.resolve("batch_" + matrixName + ".txt");
		Path designFile = outDir.resolve("matrix_design_" + matrixName + ".csv");
		if (Files.isDirectory(expDir) && hasFiles(expDir)) {
			if (!overwrite) {
				return fail(Collections.singletonList("output folder already has generated experiments: " + expDir
						+ "\n  Choose a new output.directory, or run with --overwrite to replace them."));
			}
			try (DirectoryStream<Path> ds = Files.newDirectoryStream(expDir, "*.properties")) {
				for (Path p : ds) {
					Files.delete(p);
				}
			}
		}
		Files.createDirectories(expDir);

		// ---- document lists
		Path allList = null;
		Path subsetList = null;
		if (corpusDir != null) {
			Files.createDirectories(listDir);
			List<Path> docs;
			try (Stream<Path> s = Files.list(corpusDir)) {
				docs = s.filter(p -> Files.isRegularFile(p)
						&& p.getFileName().toString().toLowerCase(Locale.ROOT).endsWith(ext.toLowerCase(Locale.ROOT)))
						.sorted(Comparator.comparing((Path p) -> p.getFileName().toString())).collect(Collectors.toList());
			}
			if (docs.isEmpty()) {
				return fail(Collections.singletonList("no *" + ext + " files in " + corpusDir));
			}
			allList = listDir.resolve("corpus_all.txt");
			writeList(allList, docs, "All " + docs.size() + " documents of " + corpusDir);
			if (subsetSize != null) {
				List<Path> sub = spreadBySize(docs, subsetSize);
				subsetList = listDir.resolve("corpus_" + subsetName + "_" + sub.size() + ".txt");
				writeList(subsetList, sub, sub.size() + " of " + docs.size()
						+ " documents spread evenly from shortest to longest (by file size)");
				System.out.println("Document lists: " + docs.size() + " documents; subset '" + subsetName + "' = "
						+ sub.size() + " documents spread across file sizes (" + sizeRange(sub) + ")");
			}
		}

		// ---- expand groups to runs
		List<Run> runs = new ArrayList<>();
		for (Group g : groups) {
			List<Run> gr = new ArrayList<>();
			expandGroup(g, slot, baseValues, machineValues, machine != null, gr, errors);
			// document list choice
			String docs = g.documents != null ? g.documents : (allList != null ? "all" : "base");
			Path docList = null;
			String shown = "base";
			if (docs.equals("base")) {
				// leave the base experiment's setting
			} else if (docs.equals("all")) {
				if (allList == null) {
					errors.add("group " + g.name + ": documents = all needs corpus.directory in the global settings");
				} else {
					docList = allList;
					shown = "all";
				}
			} else if (docs.equals(subsetName)) {
				if (subsetList == null) {
					errors.add("group " + g.name + ": documents = " + docs + " needs corpus.directory and subset.size");
				} else {
					docList = subsetList;
					shown = subsetName;
				}
			} else {
				docList = g.baseDir.resolve(docs).toAbsolutePath().normalize();
				if (!Files.isRegularFile(docList)) {
					errors.add("group " + g.name + ": document list not found: " + docList);
				}
				shown = docList.getFileName().toString();
			}
			for (Run r : gr) {
				r.documentsShown = shown;
				if (docList != null) {
					r.overrides.put("experiment.document-list", docList.toString());
				}
			}
			if (g.shuffle) {
				Collections.shuffle(gr, new Random(g.seed));
			}
			runs.addAll(gr);
		}
		if (!errors.isEmpty()) {
			return fail(errors);
		}

		// ---- write experiment files
		int width = Math.max(3, Integer.toString(runs.size()).length());
		int n = 0;
		for (Run r : runs) {
			n++;
			r.file = expDir.resolve(String.format(Locale.ROOT, "%0" + width + "d_%s.properties", n, r.label));
			Files.write(r.file, config.TrunkLocation.tokenize(experimentText(r, matrixFile, base, baseLines)).getBytes(StandardCharsets.UTF_8));
			warnAbout(r, baseValues, machineValues, warnings);
		}

		// ---- batch file and design table
		StringBuilder b = new StringBuilder();
		b.append("# Generated by ExperimentMatrixExpander from ").append(matrixFile).append(" on ")
				.append(LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm"))).append(nl());
		b.append("# Run with: RUN_EXPERIMENT_BATCH ").append(batchFile)
				.append(machine == null ? "" : " --config=" + machine).append(nl());
		if (samplerScript != null) {
			b.append(nl()).append("# Resource sampler: started when the batch starts, stopped when it ends.").append(nl());
			b.append("@sampler = ").append(samplerScript).append(nl());
			b.append("@sampler-interval = ").append(samplerInterval == null ? 5 : samplerInterval).append(nl());
		}
		String lastGroup = null;
		for (Run r : runs) {
			if (!r.group.name.equals(lastGroup)) {
				b.append(nl()).append("# ---- ").append(r.group.name);
				if (!r.group.description.isEmpty()) {
					b.append(": ").append(r.group.description);
				}
				b.append(nl());
				lastGroup = r.group.name;
			}
			b.append(r.file);
			if (r.group.repeat > 1) {
				b.append(" * ").append(r.group.repeat);
			}
			b.append(nl());
		}
		Files.write(batchFile, config.TrunkLocation.tokenize(b.toString()).getBytes(StandardCharsets.UTF_8));

		Set<String> varyKeys = new LinkedHashSet<>();
		for (Run r : runs) {
			varyKeys.addAll(r.overrides.keySet());
		}
		varyKeys.remove("experiment.document-list");
		StringBuilder d = new StringBuilder();
		d.append("order,experiment_file,label,group,repeats,documents");
		for (String k : varyKeys) {
			d.append(',').append(k);
		}
		d.append(nl());
		n = 0;
		for (Run r : runs) {
			n++;
			d.append(n).append(',').append(csv(r.file.getFileName().toString())).append(',').append(csv(r.label)).append(',')
					.append(csv(r.group.name)).append(',').append(r.group.repeat).append(',').append(csv(r.documentsShown));
			for (String k : varyKeys) {
				d.append(',').append(csv(r.overrides.getOrDefault(k, "")));
			}
			d.append(nl());
		}
		Files.write(designFile, config.TrunkLocation.tokenize(d.toString()).getBytes(StandardCharsets.UTF_8));

		// ---- summary
		int totalRuns = 0;
		double minutes = 0;
		boolean haveEstimate = false;
		System.out.println();
		System.out.printf(Locale.ROOT, "%-4s %-34s %-8s %s%n", "#", "label", "repeats", "documents");
		n = 0;
		for (Run r : runs) {
			n++;
			System.out.printf(Locale.ROOT, "%-4d %-34s %-8d %s%n", n, r.label, r.group.repeat, r.documentsShown);
			totalRuns += r.group.repeat;
			Double e = r.group.estimateMinutes != null ? r.group.estimateMinutes : estimateGlobal;
			if (e != null) {
				minutes += e * r.group.repeat;
				haveEstimate = true;
			}
		}
		System.out.println();
		System.out.println(runs.size() + " experiment file(s), " + totalRuns + " run(s) in total.");
		if (haveEstimate) {
			System.out.printf(Locale.ROOT, "Estimated time: %.1f hours (from your estimate.minutes values, plus about "
					+ "10 s per run for server start and pause).%n", (minutes + totalRuns * 10.0 / 60.0) / 60.0);
		}
		System.out.println("Experiment files : " + expDir);
		System.out.println("Batch file       : " + batchFile);
		System.out.println("Design table     : " + designFile);

		// ---- validate with the application's own parser
		int bad = 0;
		if (machine != null) {
			java.io.PrintStream realErr = System.err;
			for (Run r : runs) {
				// The settings parser only WARNS about a misspelled setting name; capture that and treat it as a problem.
				java.io.ByteArrayOutputStream captured = new java.io.ByteArrayOutputStream();
				System.setErr(new java.io.PrintStream(captured, true));
				try {
					AppSettings.fromFiles(machine, r.file);
				} catch (Exception e) {
					bad++;
					errors.add(r.file.getFileName() + ": " + String.valueOf(e.getLocalizedMessage()).replace("\n", "\n    "));
				} finally {
					System.setErr(realErr);
				}
				for (String line : captured.toString().split("\\r?\\n")) {
					if (line.contains("unknown setting")) {
						bad++;
						errors.add(r.file.getFileName() + ": " + line.replace("WARNING: ", ""));
					} else if (!line.trim().isEmpty()) {
						System.err.println(line);
					}
				}
			}
			if (bad == 0) {
				System.out.println("Validation       : all " + runs.size() + " experiment files load without errors with "
						+ machine.getFileName());
			}
		} else {
			warnings.add("machine.config is not set, so the generated files were not validated.");
		}
		for (String w : warnings) {
			System.out.println("WARNING: " + w);
		}
		if (!errors.isEmpty()) {
			return fail(errors);
		}
		System.out.println();
		System.out.println("Next: RUN_EXPERIMENT_BATCH " + batchFile + (machine == null ? "" : " --config=" + machine));
		return 0;
	}

	// ------------------------------------------------------------------ expansion

	private static void expandGroup(Group g, Integer slot, Map<String, String> baseValues,
			Map<String, String> machineValues, boolean haveMachine, List<Run> out, List<String> errors) {
		Map<String, String> fixed = new LinkedHashMap<>();
		for (String[] s : g.sets) {
			String key = s[0];
			boolean app = s[2].equals("1");
			String value = app ? appendValue(key, s[1], baseValues, machineValues, haveMachine, g, errors) : s[1];
			if (value == null) {
				continue;
			}
			if (fixed.containsKey(key)) {
				errors.add("group " + g.name + ": '" + key + "' is set twice");
			}
			fixed.put(key, value);
		}
		int[] idx = new int[g.varies.size()];
		Set<String> labels = new LinkedHashSet<>();
		while (true) {
			Map<String, String> ov = new LinkedHashMap<>(fixed);
			Map<String, String> shown = new LinkedHashMap<>();
			List<String> tokens = new ArrayList<>();
			for (int i = 0; i < g.varies.size(); i++) {
				Vary v = g.varies.get(i);
				Alt a = v.alts.get(idx[i]);
				String value = v.append ? appendValue(v.key, a.value, baseValues, machineValues, haveMachine, g, errors) : a.value;
				if (value == null) {
					return;
				}
				if (ov.containsKey(v.key)) {
					errors.add("group " + g.name + ": '" + v.key + "' is both set and varied");
				}
				ov.put(v.key, value);
				shown.put(v.key, a.alias != null ? a.alias : a.value);
				tokens.add(token(v.key, a, idx[i]));
			}
			if (slot != null && ov.containsKey("run.default-parallel") && !ov.containsKey("server.ctx-size")) {
				try {
					ov.put("server.ctx-size", Integer.toString(Integer.parseInt(ov.get("run.default-parallel").trim()) * slot));
				} catch (NumberFormatException e) {
					errors.add("group " + g.name + ": run.default-parallel must be a whole number: " + ov.get("run.default-parallel"));
				}
			}
			String label = sanitize(tokens.isEmpty() ? g.name : g.name + "_" + String.join("_", tokens));
			if (!labels.add(label)) {
				errors.add("group " + g.name + ": two runs would get the same label '" + label + "' (duplicate values?)");
			}
			Run r = new Run(g, label);
			r.overrides.putAll(ov);
			r.varied.putAll(shown);
			out.add(r);

			int k = g.varies.size() - 1;
			while (k >= 0) {
				idx[k]++;
				if (idx[k] < g.varies.get(k).alts.size()) {
					break;
				}
				idx[k] = 0;
				k--;
			}
			if (k < 0) {
				break;
			}
		}
	}

	private static String appendValue(String key, String addition, Map<String, String> baseValues,
			Map<String, String> machineValues, boolean haveMachine, Group g, List<String> errors) {
		String existing = baseValues.containsKey(key) ? baseValues.get(key) : machineValues.get(key);
		if (existing == null && !haveMachine && !key.equals("server.extra-args")) {
			errors.add("group " + g.name + ": '+" + key + "' needs machine.config (or the base experiment) to supply the existing value");
			return null;
		}
		if (existing == null && !haveMachine) {
			errors.add("group " + g.name + ": '+" + key + "' needs machine.config so the machine file's own server.extra-args are kept");
			return null;
		}
		String e = existing == null ? "" : existing.trim();
		String a = addition.trim();
		return (e + (e.isEmpty() || a.isEmpty() ? "" : " ") + a).trim();
	}

	private static String token(String key, Alt a, int index) {
		String name;
		switch (key) {
		case "run.default-parallel": name = "np"; break;
		case "server.ctx-size": name = "ctx"; break;
		case "server.threads": name = "t"; break;
		case "server.batch-size": name = "b"; break;
		case "server.ubatch-size": name = "ub"; break;
		case "server.flash-attn": name = "fa"; break;
		case "server.n-gpu-layers": name = "ngl"; break;
		case "server.extra-args": name = "x"; break;
		default: name = key.substring(key.lastIndexOf('.') + 1).replace("-", "");
		}
		if (a.alias != null) {
			return a.alias;
		}
		String v = a.value.trim();
		if (v.matches("[0-9]+(\\.[0-9]+)?")) {
			return name + v.replace('.', 'p');
		}
		String s = v.toLowerCase(Locale.ROOT).replaceAll("[^a-z0-9]", "");
		return s.length() > 0 && s.length() <= 10 ? name + s : name + (index + 1);
	}

	private static String experimentText(Run r, Path matrixFile, Path base, List<String> baseLines) {
		Set<String> replaced = new LinkedHashSet<>(r.overrides.keySet());
		replaced.add("experiment.label");
		replaced.add("experiment.notes");
		replaced.add("experiment.group");
		Path baseDir = base.getParent();
		StringBuilder s = new StringBuilder();
		s.append("# GENERATED by ExperimentMatrixExpander on ")
				.append(LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm"))).append(nl());
		s.append("# Matrix file : ").append(matrixFile).append("   group: ").append(r.group.name).append(nl());
		s.append("# Base file   : ").append(base).append(nl());
		s.append("# Do not edit by hand: change the matrix file and expand again.").append(nl()).append(nl());
		s.append("# ---- copied from the base experiment file (relative paths made absolute) ----").append(nl());
		for (String line : baseLines) {
			String t = line.replace("\uFEFF", "").trim();
			if (t.isEmpty() || t.startsWith("#") || t.startsWith("!")) {
				s.append(line.replace("\uFEFF", "")).append(nl());
				continue;
			}
			int eq = t.indexOf('=');
			if (eq < 1) {
				s.append("# ").append(t).append(nl());
				continue;
			}
			String key = t.substring(0, eq).trim();
			if (replaced.contains(key)) {
				s.append("# (replaced by the matrix) ").append(t).append(nl());
			} else if (PATH_KEYS.contains(key)) {
				String v = unquote(t.substring(eq + 1).trim());
				if (!v.isEmpty()) {
					Path p = Paths.get(v);
					v = (p.isAbsolute() ? p : baseDir.resolve(p)).normalize().toString();
				}
				s.append(key).append(" = ").append(v).append(nl());
			} else {
				s.append(t).append(nl());
			}
		}
		s.append(nl()).append("# ---- set by the matrix ----").append(nl());
		s.append("experiment.label = ").append(r.label).append(nl());
		s.append("experiment.group = ").append(r.group.name).append(nl());
		StringBuilder notes = new StringBuilder("Matrix group ").append(r.group.name);
		if (!r.varied.isEmpty()) {
			notes.append(": ");
			boolean first = true;
			for (Map.Entry<String, String> e : r.varied.entrySet()) {
				notes.append(first ? "" : "; ").append(e.getKey()).append("=").append(e.getValue());
				first = false;
			}
		}
		if (!r.group.description.isEmpty()) {
			notes.append(" (").append(r.group.description).append(")");
		}
		s.append("experiment.notes = ").append(notes.toString().replace('\n', ' ')).append(nl());
		for (Map.Entry<String, String> e : r.overrides.entrySet()) {
			s.append(e.getKey()).append(" = ").append(e.getValue()).append(nl());
		}
		return s.toString();
	}

	private static void warnAbout(Run r, Map<String, String> baseValues, Map<String, String> machineValues, List<String> warnings) {
		String extra = effective("server.extra-args", r, baseValues, machineValues);
		String fa = effective("server.flash-attn", r, baseValues, machineValues);
		if (extra != null && Pattern.compile("(--cache-type-v|-ctv)\\s+(?!f16|f32)\\S+").matcher(extra).find()
				&& "off".equalsIgnoreCase(fa == null ? "" : fa.trim())) {
			warnings.add(r.label + ": a quantized V cache normally needs flash attention on (server.flash-attn = on).");
		}
		String ctx = effective("server.ctx-size", r, baseValues, machineValues);
		String np = effective("run.default-parallel", r, baseValues, machineValues);
		try {
			if (ctx != null && np != null && Integer.parseInt(ctx.trim()) / Integer.parseInt(np.trim()) < 4096) {
				warnings.add(r.label + ": under 4096 tokens per slot; many reports may be set aside as too long.");
			}
		} catch (NumberFormatException e) {
			// the validation step reports bad numbers
		}
	}

	private static String effective(String key, Run r, Map<String, String> baseValues, Map<String, String> machineValues) {
		if (r.overrides.containsKey(key)) {
			return r.overrides.get(key);
		}
		return baseValues.containsKey(key) ? baseValues.get(key) : machineValues.get(key);
	}

	// ------------------------------------------------------------------ helpers

	private static void readLines(Path file, Deque<Path> stack, List<SrcLine> out, List<String> errors) throws IOException {
		if (stack.contains(file)) {
			errors.add("include loop at " + file);
			return;
		}
		if (!Files.isRegularFile(file)) {
			errors.add("file not found: " + file);
			return;
		}
		stack.push(file);
		List<String> raw = Files.readAllLines(file, StandardCharsets.UTF_8);
		for (int i = 0; i < raw.size(); i++) {
			String line = config.TrunkLocation.expand(raw.get(i).replace("\uFEFF", ""));
			String t = line.trim();
			if (t.startsWith("include") && t.indexOf('=') > 0 && t.substring(0, t.indexOf('=')).trim().equals("include")) {
				String p = unquote(t.substring(t.indexOf('=') + 1).trim());
				Path inc = Paths.get(p);
				if (!inc.isAbsolute()) {
					inc = file.getParent().resolve(inc);
				}
				readLines(inc.normalize(), stack, out, errors);
				out.add(new SrcLine("@end-include", file, i + 1));
			} else {
				out.add(new SrcLine(line, file, i + 1));
			}
		}
		stack.pop();
	}

	private static Path pathOf(Map<String, String> g, Map<String, SrcLine> at, String key, boolean required, List<String> errors) {
		String v = g.get(key);
		if (v == null || v.isEmpty()) {
			if (required) {
				errors.add("global setting '" + key + "' is required");
			}
			return null;
		}
		if (v.equals("~") || v.startsWith("~/") || v.startsWith("~\\")) {
			v = System.getProperty("user.home") + v.substring(1);
		}
		Path p = Paths.get(v);
		if (!p.isAbsolute()) {
			p = at.get(key).file.getParent().resolve(p);
		}
		return p.toAbsolutePath().normalize();
	}

	private static Integer intOf(Map<String, String> g, String key, int min, int max, List<String> errors) {
		String v = g.get(key);
		if (v == null) {
			return null;
		}
		try {
			int n = Integer.parseInt(v.trim());
			if (n < min || n > max) {
				throw new NumberFormatException();
			}
			return n;
		} catch (NumberFormatException e) {
			errors.add(key + " must be a whole number from " + min + " to " + max + ": " + v);
			return null;
		}
	}

	private static Map<String, String> readKeyValues(Path file) throws IOException {
		Map<String, String> m = new LinkedHashMap<>();
		for (String line : Files.readAllLines(file, StandardCharsets.UTF_8)) {
			String t = config.TrunkLocation.expand(line.replace("\uFEFF", "")).trim();
			if (t.isEmpty() || t.startsWith("#") || t.startsWith("!")) {
				continue;
			}
			int eq = t.indexOf('=');
			if (eq > 0) {
				m.put(t.substring(0, eq).trim(), unquote(t.substring(eq + 1).trim()));
			}
		}
		return m;
	}

	private static List<Path> spreadBySize(List<Path> docs, int count) throws IOException {
		if (count >= docs.size()) {
			return new ArrayList<>(docs);
		}
		List<Path> bySize = new ArrayList<>(docs);
		final Map<Path, Long> size = new LinkedHashMap<>();
		for (Path p : docs) {
			size.put(p, Files.size(p));
		}
		bySize.sort(Comparator.comparing((Path p) -> size.get(p)).thenComparing(p -> p.getFileName().toString()));
		Set<Path> pick = new LinkedHashSet<>();
		for (int i = 0; i < count; i++) {
			int at = count == 1 ? bySize.size() / 2 : (int) Math.round(i * (bySize.size() - 1) / (double) (count - 1));
			pick.add(bySize.get(at));
		}
		List<Path> out = new ArrayList<>(pick);
		out.sort(Comparator.comparing((Path p) -> p.getFileName().toString()));
		return out;
	}

	private static String sizeRange(List<Path> docs) throws IOException {
		long min = Long.MAX_VALUE;
		long max = 0;
		for (Path p : docs) {
			long s = Files.size(p);
			min = Math.min(min, s);
			max = Math.max(max, s);
		}
		return min + " to " + max + " bytes";
	}

	private static void writeList(Path file, List<Path> docs, String comment) throws IOException {
		StringBuilder s = new StringBuilder("# ").append(comment).append(nl());
		for (Path p : docs) {
			s.append(p.toAbsolutePath()).append(nl());
		}
		Files.write(file, config.TrunkLocation.tokenize(s.toString()).getBytes(StandardCharsets.UTF_8));
	}

	private static Set<String> nameList(String v) {
		Set<String> out = new LinkedHashSet<>();
		if (v != null) {
			for (String p : v.split("[,\\s]+")) {
				if (!p.trim().isEmpty()) {
					out.add(p.trim());
				}
			}
		}
		return out;
	}

	private static boolean hasFiles(Path dir) throws IOException {
		try (DirectoryStream<Path> ds = Files.newDirectoryStream(dir)) {
			return ds.iterator().hasNext();
		}
	}

	private static int fail(List<String> errors) {
		System.err.println();
		System.err.println("PROBLEMS (" + errors.size() + "):");
		for (String e : errors) {
			System.err.println("  - " + e);
		}
		return 1;
	}

	private static boolean parseBool(String v) {
		if (v.equalsIgnoreCase("true") || v.equalsIgnoreCase("yes") || v.equalsIgnoreCase("on")) {
			return true;
		}
		if (v.equalsIgnoreCase("false") || v.equalsIgnoreCase("no") || v.equalsIgnoreCase("off")) {
			return false;
		}
		throw new IllegalArgumentException(v);
	}

	private static String unquote(String v) {
		if (v.length() >= 2 && v.startsWith("\"") && v.endsWith("\"")) {
			return v.substring(1, v.length() - 1);
		}
		return v;
	}

	private static String sanitize(String s) {
		return s.replaceAll("[^A-Za-z0-9_.\\-]", "_");
	}

	private static String stripExtension(String name) {
		int dot = name.lastIndexOf('.');
		return dot > 0 ? name.substring(0, dot) : name;
	}

	private static String csv(String v) {
		if (v.indexOf(',') >= 0 || v.indexOf('"') >= 0 || v.indexOf('\n') >= 0) {
			return "\"" + v.replace("\"", "\"\"") + "\"";
		}
		return v;
	}

	private static String nl() {
		return System.lineSeparator();
	}
}
