package core;

import java.io.File;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.*;
import java.util.function.BiConsumer;
import java.util.function.Consumer;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.module.SimpleModule;

import core.NullToNoInformationDeserializer;
import core.VascularModels.FlowVelocity;
import core.VascularModels.IcaCcaRatio;
import core.VascularModels.ModelFlowVelocity;
import core.VascularModels.ModelIcaCcaRatio;
import core.VascularModels.ModelStenosis;
import core.VascularModels.ModelVascularResult;
import core.VascularModels.Stenosis;
import core.VascularModels.ValueLocation;
import core.VascularModels.VascularResult;

public class LlamaVascularExtractor {

    private static final ObjectMapper OBJECT_MAPPER = buildObjectMapper();

    private static ObjectMapper buildObjectMapper() {
        ObjectMapper mapper = new ObjectMapper();
        SimpleModule module = new SimpleModule();
        module.addDeserializer(String.class, new NullToNoInformationDeserializer());
        mapper.registerModule(module);
        return mapper;
    }

    // -------------------------------------------------------------------------
    // TOKEN BUDGET PARAMETERS (TEST THESE)
    // -------------------------------------------------------------------------
    // MAX_TOKENS is the per-request OUTPUT (generation) budget, not the context
    // window. The server's per-slot context (= --ctx-size / --parallel) must hold
    // system prompt + report + OUTPUT together. Server settings assumed:
    // --ctx-size 49152 --parallel 4  => 12288 tokens per slot. The prompt and
    // report use roughly 2.5-3.5K, so MAX_TOKENS 7168 fits (about 10.7K total),
    // and RETRY_MAX_TOKENS 8192 still fits (about 11.7K). If you lower
    // --ctx-size/--parallel headroom, lower these too; llama.cpp stops at the
    // slot's context limit and reports finish_reason=length.
    //
    // Server log analysis showed many documents need 4,300-4,800 output tokens
    // (source_text excerpts roughly double tokens per entry). With the old
    // 4096 cap those documents were truncated and regenerated (~6.5 minutes of
    // a slot each). 7168 lets the first attempt complete; the retry is now a
    // rarely used safety net. Shortening source_text (see SYSTEM_PROMPT) also
    // reduces output length and therefore time per document.
    // -------------------------------------------------------------------------
    public static final int MAX_TOKENS = 7168;
    public static final int RETRY_MAX_TOKENS = 8192;

    // Repair prompt output budget — repaired JSON is never large.
    private static final int REPAIR_MAX_TOKENS = 2048;

    // -------------------------------------------------------------------------
    // RESPONSE-FORMAT MODE (TEST THIS)
    // -------------------------------------------------------------------------
    // true  -> send a full JSON schema (constrains enum values + requires all
    //          three top-level arrays). Strongest accuracy guarantee.
    // false -> send {"type":"json_object"} (well-formed JSON only).
    //
    // Set to false if your llama.cpp build does not accept json_schema in
    // response_format (the POST will return a non-2xx error if unsupported).
    // -------------------------------------------------------------------------
    private static final boolean USE_JSON_SCHEMA = true;

    // Valid controlled-vocabulary values (used for warnings in validate()).
    // These mirror the enums in buildResponseFormat() and the repair prompt.
    private static final Set<String> VALID_VESSELS         = Set.of("ICA", "ECA", "CCA");
    private static final Set<String> VALID_LATERALITY      = Set.of("Left", "Right", "InsufficientInformation");
    private static final Set<String> VALID_VELOCITY_TYPES  = Set.of("PSV", "EDV");
    private static final Set<String> VALID_FLOW_DIRECTIONS = Set.of("anterograde", "retrograde");

    private final HttpClient httpClient = HttpClient.newHttpClient();
    private final String serverUrl;
    private final String modelName;
    private final String systemPrompt;

    public LlamaVascularExtractor(String serverUrl, String modelName, String systemPrompt) {
        this.serverUrl    = serverUrl;
        this.modelName    = modelName;
        this.systemPrompt = systemPrompt;
    }

    // -------------------------------------------------------------------------
    // Public API
    // -------------------------------------------------------------------------

    public VascularResult analyzeDocument(Path documentPath) throws Exception {
        String document = Files.readString(documentPath);

        // Primary attempt.
        String rawContent;
        try {
            rawContent = callModelForContent(buildUserMessage(document), MAX_TOKENS);
        } catch (TruncationException te) {
            // Output hit the token ceiling. A plain retry at temperature 0 would
            // reproduce identical (still-truncated) output, so the ONLY useful
            // response is a larger budget. Jump straight to the larger-budget
            // retry and skip the repair path (repair cannot fix truncation).
            System.err.println("Primary truncated for " + documentPath.getFileName()
                    + "; retrying with larger token budget.");
            String retryContent = callModelForContent(buildUserMessage(document), RETRY_MAX_TOKENS);
            ModelVascularResult retryResult =
                    OBJECT_MAPPER.readValue(retryContent.trim(), ModelVascularResult.class);
            validate(retryResult);
            return enrich(retryResult, documentPath, document);
        }

        // Primary produced complete output. Try to parse/validate it.
        try {
            ModelVascularResult modelResult = OBJECT_MAPPER.readValue(rawContent.trim(), ModelVascularResult.class);
            validate(modelResult);
            return enrich(modelResult, documentPath, document);

        } catch (Exception primaryFailure) {
            // Complete-but-malformed output. A plain retry at temperature 0 would
            // reproduce the same bad output, so go to the REPAIR path, which uses
            // a DIFFERENT prompt and can therefore change the outcome.
            //
            // NOTE: with the response_format JSON-schema constraint enabled the
            // model is mechanically prevented from emitting invalid JSON, so this
            // branch should rarely execute. It is retained as defense-in-depth.
            System.err.println("Primary parse/validate failed for "
                    + documentPath.getFileName() + ": " + primaryFailure.getMessage());

            try {
                ModelVascularResult repaired = repairInvalidJson(rawContent);
                validate(repaired);
                return enrich(repaired, documentPath, document);

            } catch (Exception repairFailure) {
                System.err.println("JSON repair failed for "
                        + documentPath.getFileName() + ": " + repairFailure.getMessage());
                throw repairFailure;
            }
        }
    }

    /**
     * Convenience overload: failures are reported to System.err and the run
     * continues. See the 4-argument version for details.
     */
    public List<VascularResult> analyzeDocumentsParallel(
            List<Path> documentPaths,
            int threads,
            Consumer<VascularResult> onResultReady) throws Exception {

        return analyzeDocumentsParallel(documentPaths, threads, onResultReady,
                (path, e) -> System.err.println(
                        "FAILED " + path.getFileName() + ": " + e.getMessage()));
    }

    /**
     * Processes documents concurrently. A failure on any single document (server
     * error, unreadable file, unrecoverable JSON, etc.) is reported through
     * onFailure and does NOT abort the remaining documents.
     *
     * @param onResultReady called once per successfully processed document
     * @param onFailure     called once per failed document with the path and the
     *                      exception; callers can use it to log and to collect a
     *                      re-run list
     * @return results for the documents that succeeded (failed documents are
     *         omitted), in submission order
     */
    public List<VascularResult> analyzeDocumentsParallel(
            List<Path> documentPaths,
            int threads,
            Consumer<VascularResult> onResultReady,
            BiConsumer<Path, Exception> onFailure) throws Exception {

        try (ExecutorService executor = Executors.newFixedThreadPool(threads)) {

            List<Future<VascularResult>> futures = new ArrayList<>();

            for (Path documentPath : documentPaths) {
                futures.add(executor.submit(() -> {
                    try {
                        // A missing/unreadable file surfaces as an IOException
                        // from Files.readString inside analyzeDocument and is
                        // handled here like any other per-document failure.
                        VascularResult result = analyzeDocument(documentPath);
                        onResultReady.accept(result);
                        return result;
                    } catch (Exception e) {
                        onFailure.accept(documentPath, e);
                        return null;
                    }
                }));
            }

            List<VascularResult> results = new ArrayList<>();
            for (Future<VascularResult> future : futures) {
                VascularResult r = future.get();
                if (r != null) {
                    results.add(r);
                }
            }
            return results;
        }
    }

    // -------------------------------------------------------------------------
    // Prompt builders
    // -------------------------------------------------------------------------

    // The user message contains ONLY the report text. The fixed rules, schema,
    // and worked example live in the system prompt (passed in from
    // MainVascularRevised.SYSTEM_PROMPT). This split follows Gemma 4's native
    // system-role support and keeps the large fixed prefix constant so that
    // cache_prompt=true can reuse its KV cache across documents.
    private String buildUserMessage(String documentText) {
        return "Now extract from this report:\n<<<\n" + documentText + "\n>>>";
    }

    // The schema in this prompt is the authoritative schema for the revised
    // (XML-aligned) output. SYSTEM_PROMPT in MainVascularRevised,
    // buildResponseFormat() below, and VascularModelsRevised must all match it.
    private static String buildJsonRepairPrompt(String invalidJson) {
        // String.replace (not formatted) so a '%' in the invalid JSON is safe.
        return ("""
                Repair the following text into valid JSON.

                Required exact schema:
                {
                  "ICA_CCA_ratio": [
                    { "laterality": "Left|Right|InsufficientInformation",
                      "source_text": "<short excerpt copied verbatim from the report that contains the value>",
                      "ratio_value": "<number as string or InsufficientInformation>" }
                  ],
                  "flow_velocities": [
                    { "vessel": "ICA|ECA|CCA",
                      "laterality": "Left|Right|InsufficientInformation",
                      "vessel_location": "proximal|mid|distal|InsufficientInformation",
                      "velocity_type": "PSV|EDV",
                      "flow_direction": "anterograde|retrograde",
                      "source_text": "<short excerpt copied verbatim from the report that contains the value>",
                      "value": "<number as string, NoMeasurableFlow, or InsufficientInformation>" }
                  ],
                  "stenosis": [
                    { "vessel": "ICA|ECA|CCA",
                      "laterality": "Left|Right|InsufficientInformation",
                      "source_text": "<short excerpt copied verbatim from the report that contains the value>",
                      "percentage": "<string or InsufficientInformation>" }
                  ]
                }

                Rules:
                - Output ONLY valid JSON.
                - Do not explain.
                - Do not use markdown.
                - Preserve apparent intended values where possible.
                - Preserve source_text exactly as given; if it is missing, use an empty string.
                - Use empty arrays [] for sections with no findings.

                Text to repair:
                <<<
                {{INVALID}}
                >>>
                """).replace("{{INVALID}}", invalidJson);
    }

    // -------------------------------------------------------------------------
    // HTTP and model interaction
    // -------------------------------------------------------------------------

    private ModelVascularResult repairInvalidJson(String invalidJson) throws Exception {
        // The repair prompt is self-contained, so it is sent as a single user
        // message with no system prompt (no cache benefit here — repairs are rare).
        String repaired = callModelForContent(List.of(
                Map.of("role", "user", "content", buildJsonRepairPrompt(invalidJson))),
                REPAIR_MAX_TOKENS);
        return OBJECT_MAPPER.readValue(repaired.trim(), ModelVascularResult.class);
    }

    // Convenience overload: builds the system + user message pair for a report.
    private String callModelForContent(String userMessage, int maxTokens) throws Exception {
        List<Map<String, String>> messages = List.of(
                Map.of("role", "system", "content", systemPrompt),
                Map.of("role", "user", "content", userMessage));
        return callModelForContent(messages, maxTokens);
    }

    private String callModelForContent(List<Map<String, String>> messages, int maxTokens) throws Exception {
        Map<String, Object> requestBody = new LinkedHashMap<>();
        requestBody.put("model", modelName);

        // Deterministic greedy decoding — correct for schema-locked extraction.
        requestBody.put("temperature", 0);
        requestBody.put("top_k", 1);
        requestBody.put("top_p", 1.0);
        requestBody.put("repeat_penalty", 1.0);
        requestBody.put("max_tokens", maxTokens);
        requestBody.put("stream", false);

        // THROUGHPUT: reuse the KV cache of the constant system-prompt prefix
        // across documents. Pairs with the system/user split above.
        requestBody.put("cache_prompt", true);

        // ACCURACY: constrain output at generation time (json_schema or
        // json_object, selected by USE_JSON_SCHEMA).
        requestBody.put("response_format", buildResponseFormat());

        requestBody.put("messages", messages);

        String responseJson = postJson(serverUrl + "/v1/chat/completions", requestBody);
        JsonNode root = OBJECT_MAPPER.readTree(responseJson);
        JsonNode contentNode = root.path("choices").path(0).path("message").path("content");

        if (contentNode.isMissingNode() || contentNode.isNull()) {
            throw new RuntimeException("No message.content in response:\n" + responseJson);
        }

        // Detect truncation before parsing — "length" means max_tokens (or the
        // slot's context limit) was hit. Thrown as a typed exception so
        // analyzeDocument can route truncation to the larger-budget retry rather
        // than the (useless-for-truncation) repair path.
        String finishReason = root.path("choices").path(0).path("finish_reason").asText("");
        if ("length".equals(finishReason)) {
            throw new TruncationException("Model response truncated (finish_reason=length). "
                    + "Partial content:\n" + contentNode.asText());
        }

        String content = contentNode.asText();
        if (content == null || content.isBlank()) {
            throw new RuntimeException("Empty model response:\n" + responseJson);
        }

        // Defensive: strip markdown code fences if the model emits them despite
        // the response_format constraint (should not occur, but harmless).
        content = content.trim();
        if (content.startsWith("```")) {
            content = content.replaceAll("^```[a-zA-Z]*\\n?", "").replaceAll("```$", "").trim();
        }

        return content;
    }

    // -------------------------------------------------------------------------
    // Response-format builder
    // -------------------------------------------------------------------------

    // Builds the response_format object for the request. When USE_JSON_SCHEMA is
    // true, returns a full JSON schema that (a) requires all three top-level
    // arrays and (b) constrains the controlled-vocabulary fields to their legal
    // enum values at generation time. When false, returns the weaker
    // {"type":"json_object"} form.
    //
    // The schema mirrors VascularModelsRevised.ModelVascularResult and the schema
    // in buildJsonRepairPrompt. Values are JSON strings throughout (matching the
    // records), using "enum" where the vocabulary is closed. ratio_value, value,
    // and percentage remain free strings because the prompt requires verbatim
    // preservation (including ranges like "60-69%").
    //
    // ACCURACY NOTE (TEST THIS): velocity_type (PSV|EDV) and flow_direction
    // (anterograde|retrograde) have NO "InsufficientInformation" escape value, as
    // in the repair-prompt schema. The model is therefore forced to pick one even
    // when the report is silent. The system prompt tells it to default
    // velocity_type to "PSV" and flow_direction to "anterograde" when the report
    // does not state them,
    // but if you see spurious values, add "InsufficientInformation" to these two
    // enums here (and in the repair prompt / system prompt / validate()).
    private static Object buildResponseFormat() {
        if (!USE_JSON_SCHEMA) {
            return Map.of("type", "json_object");
        }

        List<String> vessels       = List.of("ICA", "ECA", "CCA");
        List<String> laterality    = List.of("Left", "Right", "InsufficientInformation");
        List<String> velocityType  = List.of("PSV", "EDV");
        List<String> flowDirection = List.of("anterograde", "retrograde");
        List<String> vesselLoc     = List.of("proximal", "mid", "distal", "InsufficientInformation");

        Map<String, Object> ratioItem = objectSchema(
                orderedProps(
                        "laterality",  enumString(laterality),
                        "source_text", Map.of("type", "string"),
                        "ratio_value", Map.of("type", "string")),
                List.of("laterality", "source_text", "ratio_value"));

        Map<String, Object> velocityItem = objectSchema(
                orderedProps(
                        "vessel",          enumString(vessels),
                        "laterality",      enumString(laterality),
                        "vessel_location", enumString(vesselLoc),
                        "velocity_type",   enumString(velocityType),
                        "flow_direction",  enumString(flowDirection),
                        "source_text",     Map.of("type", "string"),
                        "value",           Map.of("type", "string")),
                List.of("vessel", "laterality", "vessel_location", "velocity_type",
                        "flow_direction", "source_text", "value"));

        Map<String, Object> stenosisItem = objectSchema(
                orderedProps(
                        "vessel",      enumString(vessels),
                        "laterality",  enumString(laterality),
                        "source_text", Map.of("type", "string"),
                        "percentage",  Map.of("type", "string")),
                List.of("vessel", "laterality", "source_text", "percentage"));

        Map<String, Object> rootProps = orderedProps(
                "ICA_CCA_ratio",   arrayOf(ratioItem),
                "flow_velocities", arrayOf(velocityItem),
                "stenosis",        arrayOf(stenosisItem));

        Map<String, Object> rootSchema = objectSchema(
                rootProps,
                List.of("ICA_CCA_ratio", "flow_velocities", "stenosis"));

        Map<String, Object> jsonSchema = new LinkedHashMap<>();
        jsonSchema.put("name", "vascular_extraction");
        jsonSchema.put("strict", true);
        jsonSchema.put("schema", rootSchema);

        Map<String, Object> responseFormat = new LinkedHashMap<>();
        responseFormat.put("type", "json_schema");
        responseFormat.put("json_schema", jsonSchema);
        return responseFormat;
    }

    // --- small schema-construction helpers ---

    private static Map<String, Object> enumString(List<String> values) {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("type", "string");
        m.put("enum", values);
        return m;
    }

    private static Map<String, Object> arrayOf(Map<String, Object> itemSchema) {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("type", "array");
        m.put("items", itemSchema);
        return m;
    }

    // Builds an object schema with additionalProperties:false so the model cannot
    // invent fields outside the schema.
    private static Map<String, Object> objectSchema(Map<String, Object> properties, List<String> required) {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("type", "object");
        m.put("properties", properties);
        m.put("required", required);
        m.put("additionalProperties", false);
        return m;
    }

    // Builds an ordered property map from alternating (name, schema) arguments.
    private static Map<String, Object> orderedProps(Object... kv) {
        if (kv.length % 2 != 0) {
            throw new IllegalArgumentException("orderedProps requires an even number of arguments");
        }
        Map<String, Object> m = new LinkedHashMap<>();
        for (int i = 0; i < kv.length; i += 2) {
            m.put((String) kv[i], kv[i + 1]);
        }
        return m;
    }

    private String postJson(String url, Object body) throws Exception {
        String json = OBJECT_MAPPER.writeValueAsString(body);
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(url))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(json))
                .build();

        HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());

        if (response.statusCode() < 200 || response.statusCode() >= 300) {
            throw new RuntimeException(
                    "llama.cpp server error: " + response.statusCode() + "\n" + response.body());
        }
        return response.body();
    }

    // Typed exception so the caller can distinguish truncation (needs a larger
    // token budget) from malformed-but-complete output (needs the repair prompt).
    private static class TruncationException extends RuntimeException {
        private static final long serialVersionUID = 1L;
        TruncationException(String message) { super(message); }
    }

    // -------------------------------------------------------------------------
    // Validation
    // -------------------------------------------------------------------------

    private static void validate(ModelVascularResult result) {
        if (result == null) {
            throw new IllegalArgumentException("Null ModelVascularResult");
        }
        if (result.ICA_CCA_ratio() == null || result.flow_velocities() == null
                || result.stenosis() == null) {
            throw new IllegalArgumentException(
                    "One or more required top-level arrays are null — likely a schema mismatch");
        }

        for (ModelIcaCcaRatio r : result.ICA_CCA_ratio()) {
            if (!VALID_LATERALITY.contains(r.laterality())) {
                System.err.println("Warning: unexpected laterality in ICA/CCA ratio: " + r.laterality());
            }
        }

        for (ModelFlowVelocity v : result.flow_velocities()) {
            if (!VALID_VESSELS.contains(v.vessel())) {
                System.err.println("Warning: unexpected vessel value: " + v.vessel());
            }
            if (!VALID_LATERALITY.contains(v.laterality())) {
                System.err.println("Warning: unexpected laterality value: " + v.laterality());
            }
            if (!VALID_VELOCITY_TYPES.contains(v.velocity_type())) {
                System.err.println("Warning: unexpected velocity_type value: " + v.velocity_type());
            }
            if (!VALID_FLOW_DIRECTIONS.contains(v.flow_direction())) {
                System.err.println("Warning: unexpected flow_direction value: " + v.flow_direction());
            }
        }

        for (ModelStenosis s : result.stenosis()) {
            if (!VALID_VESSELS.contains(s.vessel())) {
                System.err.println("Warning: unexpected vessel in stenosis: " + s.vessel());
            }
            if (!VALID_LATERALITY.contains(s.laterality())) {
                System.err.println("Warning: unexpected laterality in stenosis: " + s.laterality());
            }
        }
    }

    // -------------------------------------------------------------------------
    // Enrichment: attach document path to every record
    // -------------------------------------------------------------------------

    // Returns true when a ratio_value / value / percentage field carries no
    // usable information. Such entries are dropped rather than reported.
    // "NoInformation" is what NullToNoInformationDeserializer substitutes for a
    // JSON null or blank, so it is treated the same as "InsufficientInformation".
    // NOTE: "NoMeasurableFlow" is a real finding (occluded vessel) and is KEPT.
    private static boolean isMissing(String s) {
        return s == null
                || s.isBlank()
                || "InsufficientInformation".equalsIgnoreCase(s.trim())
                || "NoInformation".equalsIgnoreCase(s.trim());
    }

    // Removes ICA_CCA_ratio, flow_velocities, and stenosis entries whose
    // ratio_value, value, or percentage (respectively) is missing. Applied as a
    // guarantee even though the system prompt also tells the model to omit them.
    private static ModelVascularResult dropEntriesWithoutValues(ModelVascularResult model, String fileName) {
        List<ModelIcaCcaRatio> ratios = model.ICA_CCA_ratio().stream()
                .filter(r -> !isMissing(r.ratio_value()))
                .toList();
        List<ModelFlowVelocity> velocities = model.flow_velocities().stream()
                .filter(v -> !isMissing(v.value()))
                .toList();
        List<ModelStenosis> stenoses = model.stenosis().stream()
                .filter(s -> !isMissing(s.percentage()))
                .toList();

        int dropped = (model.ICA_CCA_ratio().size() - ratios.size())
                + (model.flow_velocities().size() - velocities.size())
                + (model.stenosis().size() - stenoses.size());
        if (dropped > 0) {
            System.err.println("Dropped " + dropped + " entr" + (dropped == 1 ? "y" : "ies")
                    + " with no ratio_value/value/percentage for document: " + fileName);
        }
        return new ModelVascularResult(ratios, velocities, stenoses);
    }

    private static VascularResult enrich(ModelVascularResult unfiltered, Path documentPath, String document) {
        String pathStr = documentPath.toString();
        String fileName = new File(pathStr).getName();

        System.err.println("Enriching results for document: " + fileName);
        ModelVascularResult model = dropEntriesWithoutValues(unfiltered, fileName);
        printSummary(model, fileName);

        // One ValueLocator per section so each keeps its own sequential cursor
        // (used to choose between repeated occurrences of the same excerpt).
        // Entries are located in the order the model returned them.
        ValueLocator ratioLocator    = new ValueLocator(document);
        ValueLocator velocityLocator = new ValueLocator(document);
        ValueLocator stenosisLocator = new ValueLocator(document);

        List<IcaCcaRatio> ratios = model.ICA_CCA_ratio().stream()
                .map(r -> new IcaCcaRatio(r.laterality(), r.source_text(), r.ratio_value(),
                        ratioLocator.locate(r.source_text(), r.ratio_value())))
                .toList();

        List<FlowVelocity> velocities = model.flow_velocities().stream()
                .map(v -> new FlowVelocity(
                        v.vessel(), v.laterality(), v.vessel_location(),
                        v.velocity_type(), v.flow_direction(), v.source_text(), v.value(),
                        velocityLocator.locate(v.source_text(), v.value())))
                .toList();

        List<Stenosis> stenoses = model.stenosis().stream()
                .map(s -> new Stenosis(s.vessel(), s.laterality(), s.source_text(), s.percentage(),
                        stenosisLocator.locate(s.source_text(), s.percentage())))
                .toList();

        reportLocationStats(fileName, ratios, velocities, stenoses);

        return new VascularResult(ratios, velocities, stenoses, pathStr);
    }

    // Console summary of how many values were located in the document.
    private static void reportLocationStats(String fileName, List<IcaCcaRatio> ratios,
            List<FlowVelocity> velocities, List<Stenosis> stenoses) {
        List<ValueLocation> all = new ArrayList<>();
        ratios.forEach(r -> all.add(r.value_location()));
        velocities.forEach(v -> all.add(v.value_location()));
        stenoses.forEach(s -> all.add(s.value_location()));

        long notFound  = all.stream().filter(l -> !l.found()).count();
        long ambiguous = all.stream().filter(l -> l.found() && l.ambiguous()).count();
        long snippetOnly = all.stream()
                .filter(l -> l.found() && l.match_method().endsWith("+SNIPPET_ONLY")).count();
        long valueOnly = all.stream()
                .filter(l -> l.found() && "VALUE_ONLY".equals(l.match_method())).count();

        System.err.println("Located " + (all.size() - notFound) + " of " + all.size()
                + " values in " + fileName + " (ambiguous: " + ambiguous
                + ", excerpt-only: " + snippetOnly
                + ", value-only fallback: " + valueOnly
                + ", not found: " + notFound + ")");
    }

    private static void printSummary(ModelVascularResult model, String fileName) {
        System.out.println("\n--------------------------------------------");
        System.out.println("Document: " + fileName);
        System.out.println("ICA/CCA ratios:    " + model.ICA_CCA_ratio().size());
        System.out.println("Flow velocities:   " + model.flow_velocities().size());
        System.out.println("Stenosis entries:  " + model.stenosis().size());

        model.ICA_CCA_ratio().forEach(r ->
                System.out.println("  Ratio " + r.laterality() + ": " + r.ratio_value()));

        model.flow_velocities().forEach(v ->
                System.out.println("  " + v.laterality() + " " + v.vessel()
                        + " (" + v.vessel_location() + ") "
                        + v.velocity_type() + " " + v.flow_direction() + ": " + v.value()));

        model.stenosis().forEach(s ->
                System.out.println("  Stenosis " + s.laterality() + " " + s.vessel()
                        + ": " + s.percentage()));

        System.out.println("--------------------------------------------\n");
    }
}
