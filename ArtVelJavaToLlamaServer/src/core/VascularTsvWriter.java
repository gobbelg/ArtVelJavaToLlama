package core;

import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import core.VascularModels.*;

/**
 * Writes the extraction results to four tab-delimited files next to the JSON
 * output file:
 *
 *   <base>_ICA_CCA_ratio.tsv
 *   <base>_flow_velocities.tsv
 *   <base>_stenosis.tsv
 *   <base>_documents.tsv      ONE ROW PER DOCUMENT ATTEMPT (see below)
 *
 * The first three hold ONE extracted value per line, so a document with no
 * findings leaves no trace in them. The documents file fills that gap: every
 * document gets a row with its status, so "processed, nothing found" can be told
 * apart from "failed" without reading the log.
 *
 *   status SUCCESS             processed; at least one value extracted
 *   status SUCCESS_NO_FINDINGS processed; the model found nothing to extract
 *   status FAILED              could not be processed (see the error column)
 *   status TOO_LONG            longer than the largest slot size that was used
 *   status NOT_PROCESSED       the run aborted before reaching it
 *
 * Other columns: pass (1 = first pass; a document that was too long for pass 1
 * and then succeeded shows pass 2), counts per section, and location statistics:
 * located / ambiguous / excerpt_only / value_only / not_found / reused_location
 * (see ValueLocator), max_identical_entries (how often the most repeated entry
 * occurs; 4 or more suggests a repetition loop), notes (retry, JSON repair,
 * reasoning output) and error.
 *
 * where <base> is the JSON output file name without its extension. Each line
 * holds ONE extracted value from ONE document; the document's file name (not
 * its full path) is in the first column. start_offset / end_offset / matched_text
 * come from the value_location computed by ValueLocator (-1 / -1 / empty when
 * the value could not be located). Files are opened in append mode, matching the
 * JSON output; the header line is written only when a file is new or empty.
 *
 * Tabs, carriage returns, and line feeds inside any field are written as the
 * literal two-character escapes \t, \r, and \n so every record stays on one
 * line with the correct number of columns.
 *
 * NOT thread-safe: call write() from inside the caller's write lock (as
 * MainVascularRevised does).
 */
public class VascularTsvWriter implements AutoCloseable {

    // Column names. Edit here to rename columns.
    private static final List<String> RATIO_COLUMNS = List.of(
            "DocumentName", "laterality", "ratio_value",
            "start_offset", "end_offset", "matched_text");

    private static final List<String> VELOCITY_COLUMNS = List.of(
            "DocumentName", "vessel", "laterality", "vessel_location",
            "flow_direction", "velocity_type", "value", "units",
            "start_offset", "end_offset", "matched_text");

    private static final List<String> STENOSIS_COLUMNS = List.of(
            "DocumentName", "vessel", "laterality", "percentage",
            "start_offset", "end_offset", "matched_text");

    private static final List<String> DOCUMENT_COLUMNS = List.of(
            "DocumentName", "DocumentPath", "status", "pass",
            "ICA_CCA_ratios", "flow_velocities", "stenoses", "values_total",
            "located", "ambiguous", "excerpt_only", "value_only", "not_found", "reused_location",
            "max_identical_entries", "notes", "error");

    /** Added to documents.tsv, after run_id, only when a run id is given (experiment mode). */
    private static final List<String> TELEMETRY_COLUMNS = List.of(
            "wall_ms", "request_ms", "model_calls", "prompt_tokens", "completion_tokens",
            "prompt_tokens_per_s", "generation_tokens_per_s", "cached_prompt_tokens",
            "start_epoch_ms", "end_epoch_ms");

    private static final int MAX_ERROR_CHARS = 500;

    /** Null = legacy columns; otherwise run_id (and telemetry in documents.tsv) are added. */
    private final String runId;

    private final PrintWriter ratioWriter;
    private final PrintWriter velocityWriter;
    private final PrintWriter stenosisWriter;
    private final PrintWriter documentWriter;

    private final Path ratioFile;
    private final Path velocityFile;
    private final Path stenosisFile;
    private final Path documentFile;

    public VascularTsvWriter(Path jsonOutputFile) throws IOException {
        this(jsonOutputFile, null);
    }

    /**
     * @param runId when not null (experiment mode) every file gets a final run_id column
     *              and documents.tsv also gets per-document timing and token columns.
     *              With null the columns are exactly as before, so appending to files
     *              from earlier runs still lines up.
     */
    public VascularTsvWriter(Path jsonOutputFile, String runId) throws IOException {
        this.runId = runId;
        String name = jsonOutputFile.getFileName().toString();
        int dot = name.lastIndexOf('.');
        String base = dot > 0 ? name.substring(0, dot) : name;
        Path dir = jsonOutputFile.toAbsolutePath().getParent();

        ratioFile    = dir.resolve(base + "_ICA_CCA_ratio.tsv");
        velocityFile = dir.resolve(base + "_flow_velocities.tsv");
        stenosisFile = dir.resolve(base + "_stenosis.tsv");
        documentFile = dir.resolve(base + "_documents.tsv");

        ratioWriter    = open(ratioFile, withRunId(RATIO_COLUMNS, false));
        velocityWriter = open(velocityFile, withRunId(VELOCITY_COLUMNS, false));
        stenosisWriter = open(stenosisFile, withRunId(STENOSIS_COLUMNS, false));
        documentWriter = open(documentFile, withRunId(DOCUMENT_COLUMNS, true));
    }

    public List<Path> getFiles() {
        return List.of(ratioFile, velocityFile, stenosisFile, documentFile);
    }

    private List<String> withRunId(List<String> columns, boolean telemetry) {
        if (runId == null) {
            return columns;
        }
        List<String> all = new java.util.ArrayList<>(columns);
        all.add("run_id");
        if (telemetry) {
            all.addAll(TELEMETRY_COLUMNS);
        }
        return all;
    }

    private static PrintWriter open(Path file, List<String> columns) throws IOException {
        // Decide about the header BEFORE opening, since append mode creates the file.
        boolean needsHeader = !Files.exists(file) || Files.size(file) == 0;

        PrintWriter writer = new PrintWriter(new BufferedWriter(
                new OutputStreamWriter(new FileOutputStream(file.toFile(), true), StandardCharsets.UTF_8)));
        if (needsHeader) {
            writer.print(String.join("\t", columns) + "\n");
            writer.flush();
        }
        return writer;
    }

    /** Writes one line per extracted value plus the document row (pass 0, no notes). */
    public void write(VascularResult result) {
        write(result, 0, "");
    }

    /**
     * Writes one line per extracted value in the result, plus one row in the
     * documents file, then flushes.
     *
     * @param pass  the pass (1, 2, ...) in which the document was processed
     * @param notes diagnostic notes for the document (may be null or empty)
     */
    public void write(VascularResult result, int pass, String notes) {
        write(result, pass, notes, null);
    }

    /** As above, with the document's timing and token telemetry (used only when a run id was given). */
    public void write(VascularResult result, int pass, String notes, DocumentTelemetry telemetry) {
        String docName = Paths.get(result.documentPathString()).getFileName().toString();

        for (IcaCcaRatio r : result.ICA_CCA_ratio()) {
            ValueLocation loc = r.value_location();
            ratioWriter.print(rowWithRun(docName, r.laterality(), r.ratio_value(),
                    start(loc), end(loc), matched(loc)));
        }

        for (FlowVelocity v : result.flow_velocities()) {
            ValueLocation loc = v.value_location();
            velocityWriter.print(rowWithRun(docName, v.vessel(), v.laterality(), v.vessel_location(),
                    v.flow_direction(), v.velocity_type(), v.value(), v.units(),
                    start(loc), end(loc), matched(loc)));
        }

        for (Stenosis s : result.stenosis()) {
            ValueLocation loc = s.value_location();
            stenosisWriter.print(rowWithRun(docName, s.vessel(), s.laterality(), s.percentage(),
                    start(loc), end(loc), matched(loc)));
        }

        writeDocumentRow(result, docName, pass, notes, telemetry);

        ratioWriter.flush();
        velocityWriter.flush();
        stenosisWriter.flush();
        documentWriter.flush();
    }

    /**
     * Writes a documents-file row for a document that produced no result
     * (status FAILED, TOO_LONG or NOT_PROCESSED).
     *
     * @param documentPath the document's path
     * @param error        a message (only the first 500 characters are kept)
     */
    public void writeStatus(String documentPath, String status, int pass, String notes, String error) {
        writeStatus(documentPath, status, pass, notes, error, null);
    }

    public void writeStatus(String documentPath, String status, int pass, String notes, String error,
            DocumentTelemetry telemetry) {
        String name = Paths.get(documentPath).getFileName().toString();
        String shortError = error == null ? "" : error;
        if (shortError.length() > MAX_ERROR_CHARS) {
            shortError = shortError.substring(0, MAX_ERROR_CHARS) + "...";
        }
        documentWriter.print(documentRow(telemetry, name, documentPath, status, Integer.toString(pass),
                "", "", "", "", "", "", "", "", "", "", "", notes, shortError));
        documentWriter.flush();
    }

    private void writeDocumentRow(VascularResult result, String docName, int pass, String notes,
            DocumentTelemetry telemetry) {
        List<ValueLocation> locations = new java.util.ArrayList<>();
        result.ICA_CCA_ratio().forEach(r -> locations.add(r.value_location()));
        result.flow_velocities().forEach(v -> locations.add(v.value_location()));
        result.stenosis().forEach(x -> locations.add(x.value_location()));

        int total = locations.size();
        long notFound = 0, ambiguous = 0, excerptOnly = 0, valueOnly = 0, reused = 0;
        for (ValueLocation l : locations) {
            if (l == null || !l.found()) {
                notFound++;
                continue;
            }
            String method = l.match_method() == null ? "" : l.match_method();
            if (l.ambiguous()) {
                ambiguous++;
            }
            if (method.contains("+SNIPPET_ONLY")) {
                excerptOnly++;
            }
            if (method.startsWith("VALUE_ONLY")) {
                valueOnly++;
            }
            if (method.endsWith("+REUSED")) {
                reused++;
            }
        }

        documentWriter.print(documentRow(telemetry, docName, result.documentPathString(),
                total == 0 ? "SUCCESS_NO_FINDINGS" : "SUCCESS", Integer.toString(pass),
                Integer.toString(result.ICA_CCA_ratio().size()),
                Integer.toString(result.flow_velocities().size()),
                Integer.toString(result.stenosis().size()),
                Integer.toString(total),
                Long.toString(total - notFound), Long.toString(ambiguous),
                Long.toString(excerptOnly), Long.toString(valueOnly),
                Long.toString(notFound), Long.toString(reused),
                Integer.toString(maxIdenticalEntries(result)),
                notes, ""));
    }

    /** A row for the ratio / velocity / stenosis files: the fields plus run_id when in use. */
    private String rowWithRun(String... fields) {
        if (runId == null) {
            return row(fields);
        }
        String[] all = java.util.Arrays.copyOf(fields, fields.length + 1);
        all[fields.length] = runId;
        return row(all);
    }

    /** A documents-file row: the fields plus, when in use, run_id and the telemetry columns. */
    private String documentRow(DocumentTelemetry t, String... fields) {
        if (runId == null) {
            return row(fields);
        }
        List<String> all = new java.util.ArrayList<>(java.util.Arrays.asList(fields));
        all.add(runId);
        if (t == null) {
            for (int i = 0; i < TELEMETRY_COLUMNS.size(); i++) {
                all.add("");
            }
        } else {
            all.add(Long.toString(t.wallMs));
            all.add(Long.toString(t.requestMs));
            all.add(Integer.toString(t.modelCalls));
            all.add(Long.toString(t.promptTokens));
            all.add(Long.toString(t.completionTokens));
            all.add(rate(t.promptTokensPerSecond()));
            all.add(rate(t.generationTokensPerSecond()));
            all.add(Long.toString(t.cachedTokens));
            all.add(Long.toString(t.startEpochMs));
            all.add(Long.toString(t.endEpochMs));
        }
        return row(all.toArray(new String[0]));
    }

    private static String rate(double v) {
        return v < 0 ? "" : String.format(java.util.Locale.ROOT, "%.2f", v);
    }

    // Highest number of identical entries (ignoring their locations) in one section.
    private static int maxIdenticalEntries(VascularResult result) {
        int max = 0;
        Map<String, Integer> counts = new HashMap<>();
        for (IcaCcaRatio r : result.ICA_CCA_ratio()) {
            max = Math.max(max, counts.merge("R|" + String.join("|", nz(r.laterality()),
                    nz(r.source_text()), nz(r.ratio_value())), 1, Integer::sum));
        }
        for (FlowVelocity v : result.flow_velocities()) {
            max = Math.max(max, counts.merge("V|" + String.join("|", nz(v.vessel()),
                    nz(v.laterality()), nz(v.vessel_location()), nz(v.velocity_type()),
                    nz(v.flow_direction()), nz(v.source_text()), nz(v.value()), nz(v.units())),
                    1, Integer::sum));
        }
        for (Stenosis s : result.stenosis()) {
            max = Math.max(max, counts.merge("S|" + String.join("|", nz(s.vessel()),
                    nz(s.laterality()), nz(s.source_text()), nz(s.percentage())), 1, Integer::sum));
        }
        return max;
    }

    private static String nz(String s) {
        return s == null ? "" : s;
    }

    @Override
    public void close() {
        ratioWriter.close();
        velocityWriter.close();
        stenosisWriter.close();
        documentWriter.close();
    }

    // -------------------------------------------------------------------------

    private static String start(ValueLocation loc) {
        return loc == null ? "-1" : Integer.toString(loc.start_offset());
    }

    private static String end(ValueLocation loc) {
        return loc == null ? "-1" : Integer.toString(loc.end_offset());
    }

    private static String matched(ValueLocation loc) {
        return loc == null ? "" : loc.matched_text();
    }

    private static String row(String... fields) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < fields.length; i++) {
            if (i > 0) {
                sb.append('\t');
            }
            sb.append(clean(fields[i]));
        }
        return sb.append('\n').toString();
    }

    // Keeps each record on one line with the right number of columns.
    private static String clean(String s) {
        if (s == null) {
            return "";
        }
        return s.replace("\t", "\\t").replace("\r", "\\r").replace("\n", "\\n");
    }
}
