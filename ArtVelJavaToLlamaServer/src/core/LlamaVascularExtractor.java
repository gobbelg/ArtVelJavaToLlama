package core;

import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.charset.CharacterCodingException;
import java.nio.charset.Charset;
import java.nio.charset.CharsetDecoder;
import java.nio.charset.CodingErrorAction;
import java.nio.charset.StandardCharsets;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.net.http.HttpTimeoutException;
import java.time.Duration;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.*;
import java.util.function.BiConsumer;
import java.util.function.Consumer;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

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
    // OUTPUT TOKEN BUDGET (TUNE FROM OBSERVED DATA)
    // -------------------------------------------------------------------------
    // max_tokens is enforced by the server, not by the model: when the count is
    // reached, generation simply stops and the JSON is cut off mid-way (finish_reason
    // = length). The model does not shorten its answer to fit. Therefore:
    //
    //   - MAX_TOKENS must be ABOVE the largest legitimate output expected. It is
    //     also a runaway guard: a model stuck in a repetition loop is stopped at
    //     this cap instead of filling the rest of the slot.
    //   - ContextBudget reserves MAX_TOKENS of every slot for output and rejects
    //     (DocumentTooLongException) any report too long to leave that room. Those
    //     reports are collected for a later pass with fewer, larger slots, where
    //     the same check admits them.
    //   - Each slot's size is --ctx-size / --parallel (see LlamaServerConfig); it is
    //     passed to the constructor, not hard-coded here.
    //
    // Calibration: after every pass the extractor prints "Observed token usage"
    // with the largest completion_tokens actually produced. Set MAX_TOKENS to
    // roughly that value plus 15-25% headroom. Raising it admits fewer documents
    // in the first pass; lowering it admits more but risks truncating legitimate
    // outputs. (The previous 4,300-4,800 token observations pre-date the units field
    // and any change to source_text length, so re-measure.)
    // -------------------------------------------------------------------------
    public static final int MAX_TOKENS = 6000;

    // Larger budget for the single retry after a truncated first attempt. The retry
    // is also limited to the output room actually left in the slot, and is skipped
    // when that is not larger than MAX_TOKENS (a repeat would fail the same way).
    public static final int RETRY_MAX_TOKENS = 8192;

    // Repair prompt output budget — repaired JSON is never large.
    private static final int REPAIR_MAX_TOKENS = 2048;

    // -------------------------------------------------------------------------
    // HTTP TIMEOUTS
    // -------------------------------------------------------------------------
    // CONNECT_TIMEOUT: time allowed to open the connection to llama-server.
    // REQUEST_TIMEOUT: time allowed for the WHOLE request (prefill + generation).
    // Without it a stalled server would hang its worker thread, and therefore
    // analyzeDocumentsParallel, forever. A request that times out fails that one
    // document (it lands in the _failed list) and the run continues.
    // Choose it well above the slowest legitimate request: up to 8,192 generated
    // tokens at roughly 10 tokens/second per slot is about 14 minutes.
    // -------------------------------------------------------------------------
    private static final Duration CONNECT_TIMEOUT = Duration.ofSeconds(30);
    public static final Duration REQUEST_TIMEOUT = Duration.ofMinutes(20);

    // -------------------------------------------------------------------------
    // THINKING / REASONING (Gemma 4)
    // -------------------------------------------------------------------------
    // Gemma 4 can emit a "thought channel" before its answer. Extraction does not
    // need it and it costs generation time. DISABLE_THINKING sends
    // chat_template_kwargs {"enable_thinking": false} with every request.
    //
    // KNOWN llama.cpp ISSUE (reported on builds around 11146-11429, ggml-org/llama.cpp
    // issue 30256): with response_format json_schema, the grammar built by the
    // server still allows the thought channel even when thinking is disabled, and a
    // long prompt can make the model open it and never close it, running to
    // max_tokens. This class detects that case (reasoning_content in the response, or
    // finish_reason=length with no answer), reports it in the "Observed token usage"
    // summary and the per-document notes, and does not waste a retry on it.
    //
    // If the summary reports reasoning output, set USE_GRAMMAR_INSTEAD_OF_SCHEMA to
    // true: the same output structure is then sent as a GBNF "grammar" (the
    // documented workaround), which does not allow the thought channel.
    // -------------------------------------------------------------------------
    private static final boolean DISABLE_THINKING = true;
    private static final boolean USE_GRAMMAR_INSTEAD_OF_SCHEMA = false;

    // This many IDENTICAL entries (same vessel, side, location, type, direction,
    // value, units and excerpt) in one section is treated as a probable repetition
    // loop. Genuine duplicates are KEPT (never collapsed); they are only reported.
    private static final int LOOP_REPEAT_THRESHOLD = 4;

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
    private static final Set<String> VALID_VELOCITY_TYPES  = Set.of("PSV", "EDV", "Unspecified");
    private static final Set<String> VALID_FLOW_DIRECTIONS = Set.of("anterograde", "retrograde", "Unspecified");
    private static final Set<String> VALID_VESSEL_LOCATIONS =
            Set.of("proximal", "mid", "distal", "bulb", "bifurcation", "InsufficientInformation");
    private static final Set<String> VALID_UNITS = Set.of("cm/s", "m/s", "Other", "Unspecified");

    // HTTP/1.1 is pinned because llama-server does not speak HTTP/2; the default
    // client would first attempt an h2c upgrade on every connection.
    private final HttpClient httpClient = HttpClient.newBuilder()
            .version(HttpClient.Version.HTTP_1_1)
            .connectTimeout(CONNECT_TIMEOUT)
            .build();

    // Per-document diagnostic notes (retry used, repair used, reasoning output,
    // repeated entries). A document is processed start to finish on one thread, so
    // a ThreadLocal collects the notes during the call; they are then parked here
    // until the caller collects them with takeNotes().
    private static final ThreadLocal<List<String>> CURRENT_NOTES = new ThreadLocal<>();
    private final ConcurrentHashMap<String, String> notesByDocument = new ConcurrentHashMap<>();
    private final String serverUrl;
    private final String modelName;
    private final String systemPrompt;
    private final ContextBudget budget;
    private final UsageStats usage = new UsageStats();

    /**
     * @param slotSizeTokens context available to ONE request on the server
     *                       (--ctx-size / --parallel); see LlamaServerConfig.slotSize()
     */
    public LlamaVascularExtractor(String serverUrl, String modelName, String systemPrompt,
            int slotSizeTokens) {
        this.serverUrl    = serverUrl;
        this.modelName    = modelName;
        this.systemPrompt = systemPrompt;
        this.budget       = new ContextBudget(slotSizeTokens, systemPrompt, MAX_TOKENS);
    }

    public ContextBudget getBudget() {
        return budget;
    }

    /** Summary of real token usage reported by the server (for calibrating MAX_TOKENS). */
    public String usageSummary() {
        return usage.summary(MAX_TOKENS);
    }

    // -------------------------------------------------------------------------
    // Public API
    // -------------------------------------------------------------------------

    // -------------------------------------------------------------------------
    // Document reading with charset fallback
    // -------------------------------------------------------------------------

    private static final Charset WINDOWS_1252 = Charset.forName("windows-1252");

    /**
     * Reads a report file as text. Files.readString() assumes UTF-8 and throws
     * MalformedInputException on anything else (e.g. a Windows-1252 degree sign
     * or smart quote in a VA text export), which would fail the whole document.
     *
     * The bytes are decoded STRICTLY (malformed input is an error, never silently
     * replaced) with this fallback chain:
     *   1. the charset indicated by a byte-order mark, if the file has one
     *      (UTF-8, UTF-16LE, UTF-16BE); the BOM itself is not part of the text
     *   2. UTF-8
     *   3. Windows-1252 (the usual encoding of non-UTF-8 text from Windows systems)
     *   4. ISO-8859-1, which maps every byte value and therefore never fails
     *
     * A warning naming the file and the charset used is written to System.err
     * whenever the file is not plain UTF-8. The returned String is the single
     * text used for BOTH the model prompt and ValueLocator, so the offsets in the
     * output are offsets into this decoded text (BOM excluded).
     */
    static String readDocumentText(Path documentPath) throws IOException {
        byte[] bytes = Files.readAllBytes(documentPath);

        Charset bomCharset = null;
        int bomLength = 0;
        if (bytes.length >= 3 && (bytes[0] & 0xFF) == 0xEF
                && (bytes[1] & 0xFF) == 0xBB && (bytes[2] & 0xFF) == 0xBF) {
            bomCharset = StandardCharsets.UTF_8;
            bomLength = 3;
        } else if (bytes.length >= 2 && (bytes[0] & 0xFF) == 0xFF && (bytes[1] & 0xFF) == 0xFE) {
            bomCharset = StandardCharsets.UTF_16LE;
            bomLength = 2;
        } else if (bytes.length >= 2 && (bytes[0] & 0xFF) == 0xFE && (bytes[1] & 0xFF) == 0xFF) {
            bomCharset = StandardCharsets.UTF_16BE;
            bomLength = 2;
        }

        if (bomCharset != null) {
            String text = decodeStrict(bytes, bomLength, bytes.length - bomLength, bomCharset);
            if (text != null) {
                if (!StandardCharsets.UTF_8.equals(bomCharset)) {
                    System.err.println("Note: " + documentPath.getFileName() + " decoded as "
                            + bomCharset.name() + " (byte-order mark).");
                }
                return text;
            }
        }

        String text = decodeStrict(bytes, 0, bytes.length, StandardCharsets.UTF_8);
        if (text != null) {
            return text;
        }

        text = decodeStrict(bytes, 0, bytes.length, WINDOWS_1252);
        if (text != null) {
            System.err.println("Warning: " + documentPath.getFileName()
                    + " is not valid UTF-8; decoded as Windows-1252.");
            return text;
        }

        System.err.println("Warning: " + documentPath.getFileName()
                + " is not valid UTF-8 or Windows-1252; decoded as ISO-8859-1.");
        return new String(bytes, StandardCharsets.ISO_8859_1);
    }

    // Strict decode: returns null (instead of substituting U+FFFD) if any byte
    // sequence is malformed or unmappable in the given charset.
    private static String decodeStrict(byte[] bytes, int offset, int length, Charset charset) {
        CharsetDecoder decoder = charset.newDecoder()
                .onMalformedInput(CodingErrorAction.REPORT)
                .onUnmappableCharacter(CodingErrorAction.REPORT);
        try {
            return decoder.decode(ByteBuffer.wrap(bytes, offset, length)).toString();
        } catch (CharacterCodingException e) {
            return null;
        }
    }

    /**
     * Returns (and clears) the diagnostic notes recorded while processing the
     * given document, e.g. "RETRIED_LARGER_BUDGET; REASONING_CONTENT(1200 chars)".
     * Empty string if there were none. Works for failed documents too.
     *
     * @param documentPathString the same string as VascularResult.documentPathString()
     *                           (i.e. Path.toString() of the path passed in)
     */
    public String takeNotes(String documentPathString) {
        String notes = notesByDocument.remove(documentPathString);
        return notes == null ? "" : notes;
    }

    private static void note(String text) {
        List<String> notes = CURRENT_NOTES.get();
        if (notes != null) {
            notes.add(text);
        }
    }

    public VascularResult analyzeDocument(Path documentPath) throws Exception {
        List<String> notes = new ArrayList<>();
        CURRENT_NOTES.set(notes);
        try {
            return analyzeDocumentCore(documentPath);
        } finally {
            CURRENT_NOTES.remove();
            if (!notes.isEmpty()) {
                notesByDocument.put(documentPath.toString(), String.join("; ", notes));
            }
        }
    }

    private VascularResult analyzeDocumentCore(Path documentPath) throws Exception {
        String document = readDocumentText(documentPath);

        // Context-fit guard: reject reports too long for one slot BEFORE any model
        // call (throws ContextBudget.DocumentTooLongException; callers set these
        // aside for a pass with larger slots).
        budget.requireFits(document);

        // Primary attempt.
        String rawContent;
        try {
            rawContent = callModelForContent(buildUserMessage(document), MAX_TOKENS);
        } catch (TruncationException te) {
            // A repetition loop or an unclosed thought channel is NOT cured by a
            // larger budget (it would just run longer), so fail fast instead.
            if (te.repetitionSuspected) {
                throw new RuntimeException("Output truncated at max_tokens=" + MAX_TOKENS
                        + " and the partial output repeats the same entries (repetition loop); "
                        + "not retried with a larger budget.", te);
            }
            if (te.reasoningChars > 0 && te.contentChars == 0) {
                throw new RuntimeException("Output truncated at max_tokens=" + MAX_TOKENS
                        + " while the model was still in its thought channel (" + te.reasoningChars
                        + " characters of reasoning, no answer). Not retried. See the thinking notes "
                        + "in LlamaVascularExtractor (DISABLE_THINKING, USE_GRAMMAR_INSTEAD_OF_SCHEMA).", te);
            }

            // Output hit the token ceiling. A plain retry at temperature 0 would
            // reproduce identical (still-truncated) output, so the ONLY useful
            // response is a larger budget. Jump straight to the larger-budget
            // retry and skip the repair path (repair cannot fix truncation).
            int retryTokens = Math.min(RETRY_MAX_TOKENS, budget.availableOutputTokens(document));
            if (retryTokens <= MAX_TOKENS) {
                throw new RuntimeException("Output truncated at max_tokens=" + MAX_TOKENS
                        + " and the slot has no larger output budget for this document (about "
                        + budget.availableOutputTokens(document) + " tokens available).", te);
            }
            note("RETRIED_LARGER_BUDGET");
            System.err.println("Primary truncated for " + documentPath.getFileName()
                    + "; retrying with max_tokens=" + retryTokens + ".");
            String retryContent = callModelForContent(buildUserMessage(document), retryTokens);
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
                note("JSON_REPAIRED");
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
                        // from readDocumentText inside analyzeDocument and is
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
                      "source_text": "<excerpt copied verbatim from the report that contains the value and occurs only once in the report>",
                      "ratio_value": "<number as string or InsufficientInformation>" }
                  ],
                  "flow_velocities": [
                    { "vessel": "ICA|ECA|CCA",
                      "laterality": "Left|Right|InsufficientInformation",
                      "vessel_location": "proximal|mid|distal|bulb|bifurcation|InsufficientInformation",
                      "velocity_type": "PSV|EDV|Unspecified",
                      "flow_direction": "anterograde|retrograde|Unspecified",
                      "source_text": "<excerpt copied verbatim from the report that contains the value and occurs only once in the report>",
                      "value": "<number as string, NoMeasurableFlow, or InsufficientInformation>",
                      "units": "cm/s|m/s|Other|Unspecified" }
                  ],
                  "stenosis": [
                    { "vessel": "ICA|ECA|CCA",
                      "laterality": "Left|Right|InsufficientInformation",
                      "source_text": "<excerpt copied verbatim from the report that contains the value and occurs only once in the report>",
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
        // (or the equivalent GBNF grammar; see USE_GRAMMAR_INSTEAD_OF_SCHEMA).
        if (USE_GRAMMAR_INSTEAD_OF_SCHEMA) {
            requestBody.put("grammar", buildGrammar());
        } else {
            requestBody.put("response_format", buildResponseFormat());
        }

        // Extraction needs no reasoning; see DISABLE_THINKING.
        if (DISABLE_THINKING) {
            requestBody.put("chat_template_kwargs", Map.of("enable_thinking", false));
        }

        requestBody.put("messages", messages);

        String responseJson = postJson(serverUrl + "/v1/chat/completions", requestBody);
        JsonNode root = OBJECT_MAPPER.readTree(responseJson);
        recordUsage(root, messages);
        JsonNode contentNode = root.path("choices").path(0).path("message").path("content");

        if (contentNode.isMissingNode() || contentNode.isNull()) {
            throw new RuntimeException("No message.content in response:\n" + responseJson);
        }

        // Detect truncation before parsing — "length" means max_tokens (or the
        // slot's context limit) was hit. Thrown as a typed exception so
        // analyzeDocument can route truncation to the larger-budget retry rather
        // than the (useless-for-truncation) repair path.
        String reasoning = root.path("choices").path(0).path("message")
                .path("reasoning_content").asText("");
        if (!reasoning.isEmpty()) {
            usage.recordReasoning(reasoning.length());
            note("REASONING_CONTENT(" + reasoning.length() + " chars)");
        }

        String finishReason = root.path("choices").path(0).path("finish_reason").asText("");
        if ("length".equals(finishReason)) {
            String partial = contentNode.asText();
            boolean loop = looksLikeRepetitionLoop(partial);
            if (loop) {
                note("REPETITION_LOOP");
            }
            throw new TruncationException("Model response truncated (finish_reason=length). "
                    + "Partial content:\n" + partial, loop, reasoning.length(), partial.length());
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

    // Records the server-reported token counts (usage.prompt_tokens /
    // usage.completion_tokens) so MAX_TOKENS and ContextBudget.CHARS_PER_TOKEN can
    // be checked against real data. Called before the truncation check so that
    // truncated calls are counted too.
    private void recordUsage(JsonNode root, List<Map<String, String>> messages) {
        JsonNode u = root.path("usage");
        int promptTokens = u.path("prompt_tokens").asInt(-1);
        int completionTokens = u.path("completion_tokens").asInt(-1);
        if (promptTokens <= 0 || completionTokens < 0) {
            return;
        }
        long chars = 0;
        for (Map<String, String> m : messages) {
            chars += m.get("content").length();
        }
        usage.record(promptTokens, completionTokens, chars);
    }

    // Thread-safe accumulator for the observed token usage.
    private static final class UsageStats {
        private int calls;
        private int maxPrompt;
        private int maxCompletion;
        private double minCharsPerToken = Double.MAX_VALUE;
        private int reasoningCalls;
        private int maxReasoningChars;

        synchronized void recordReasoning(int chars) {
            reasoningCalls++;
            maxReasoningChars = Math.max(maxReasoningChars, chars);
        }

        synchronized void record(int promptTokens, int completionTokens, long promptChars) {
            calls++;
            maxPrompt = Math.max(maxPrompt, promptTokens);
            maxCompletion = Math.max(maxCompletion, completionTokens);
            minCharsPerToken = Math.min(minCharsPerToken, promptChars / (double) promptTokens);
        }

        synchronized String summary(int maxTokensCap) {
            if (calls == 0) {
                return "Observed token usage: no model calls recorded.";
            }
            StringBuilder sb = new StringBuilder();
            sb.append(String.format(
                    "Observed token usage over %d model calls: largest prompt %d tokens; "
                    + "largest completion %d tokens (MAX_TOKENS = %d); lowest characters per token %.2f "
                    + "(ContextBudget assumes %.1f; includes chat-template tokens, so slightly pessimistic).",
                    calls, maxPrompt, maxCompletion, maxTokensCap, minCharsPerToken,
                    ContextBudget.CHARS_PER_TOKEN));
            if (maxCompletion >= 0.9 * maxTokensCap) {
                sb.append("\n  NOTE: completions reached 90% or more of MAX_TOKENS; consider raising it, "
                        + "or check the long outputs for repetition loops.");
            }
            if (reasoningCalls > 0) {
                sb.append(String.format("\n  WARNING: %d model call(s) returned reasoning_content (longest %d "
                        + "characters) even though thinking is disabled. This costs generation time and can "
                        + "run to max_tokens. Known llama.cpp issue with response_format json_schema on "
                        + "Gemma 4; set USE_GRAMMAR_INSTEAD_OF_SCHEMA = true in LlamaVascularExtractor.",
                        reasoningCalls, maxReasoningChars));
            }
            if (minCharsPerToken < ContextBudget.CHARS_PER_TOKEN) {
                sb.append("\n  NOTE: some prompts were denser than the assumed characters per token; "
                        + "consider lowering ContextBudget.CHARS_PER_TOKEN.");
            }
            return sb.toString();
        }
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
    // NOTE: velocity_type (PSV|EDV|Unspecified) and flow_direction
    // (anterograde|retrograde|Unspecified) each include an "Unspecified" value so
    // the model is never forced to guess when the report is silent. units
    // (cm/s|m/s|Other|Unspecified) is likewise closed. vessel_location also allows
    // "bulb" and "bifurcation". Keep these enums in sync with the system prompt,
    // the repair prompt, and the VALID_* sets used by validate().
    private static Object buildResponseFormat() {
        if (!USE_JSON_SCHEMA) {
            return Map.of("type", "json_object");
        }

        List<String> vessels       = List.of("ICA", "ECA", "CCA");
        List<String> laterality    = List.of("Left", "Right", "InsufficientInformation");
        List<String> velocityType  = List.of("PSV", "EDV", "Unspecified");
        List<String> flowDirection = List.of("anterograde", "retrograde", "Unspecified");
        List<String> vesselLoc     = List.of("proximal", "mid", "distal", "bulb", "bifurcation", "InsufficientInformation");
        List<String> unitsEnum     = List.of("cm/s", "m/s", "Other", "Unspecified");

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
                        "value",           Map.of("type", "string"),
                        "units",           enumString(unitsEnum)),
                List.of("vessel", "laterality", "vessel_location", "velocity_type",
                        "flow_direction", "source_text", "value", "units"));

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

    // -------------------------------------------------------------------------
    // GBNF grammar (alternative to response_format json_schema)
    // -------------------------------------------------------------------------
    // Describes exactly the same output as buildResponseFormat(): three arrays in
    // a fixed order, each item an object with its fields in a fixed order, closed
    // vocabularies as alternatives, free text as JSON strings. Used only when
    // USE_GRAMMAR_INSTEAD_OF_SCHEMA is true. Keep the vocabularies in sync with
    // buildResponseFormat().
    static String buildGrammar() {
        List<String> vessels       = List.of("ICA", "ECA", "CCA");
        List<String> laterality    = List.of("Left", "Right", "InsufficientInformation");
        List<String> velocityType  = List.of("PSV", "EDV", "Unspecified");
        List<String> flowDirection = List.of("anterograde", "retrograde", "Unspecified");
        List<String> vesselLoc     = List.of("proximal", "mid", "distal", "bulb", "bifurcation",
                "InsufficientInformation");
        List<String> unitsEnum     = List.of("cm/s", "m/s", "Other", "Unspecified");

        StringBuilder g = new StringBuilder();
        g.append("root ::= ").append(gbnfObject(
                gbnfField("ICA_CCA_ratio", "ratio-array"),
                gbnfField("flow_velocities", "velocity-array"),
                gbnfField("stenosis", "stenosis-array"))).append("\n");

        g.append("ratio-array ::= ").append(gbnfArray("ratio-item")).append("\n");
        g.append("velocity-array ::= ").append(gbnfArray("velocity-item")).append("\n");
        g.append("stenosis-array ::= ").append(gbnfArray("stenosis-item")).append("\n");

        g.append("ratio-item ::= ").append(gbnfObject(
                gbnfField("laterality", "laterality"),
                gbnfField("source_text", "string"),
                gbnfField("ratio_value", "string"))).append("\n");

        g.append("velocity-item ::= ").append(gbnfObject(
                gbnfField("vessel", "vessel"),
                gbnfField("laterality", "laterality"),
                gbnfField("vessel_location", "vessel-location"),
                gbnfField("velocity_type", "velocity-type"),
                gbnfField("flow_direction", "flow-direction"),
                gbnfField("source_text", "string"),
                gbnfField("value", "string"),
                gbnfField("units", "units"))).append("\n");

        g.append("stenosis-item ::= ").append(gbnfObject(
                gbnfField("vessel", "vessel"),
                gbnfField("laterality", "laterality"),
                gbnfField("source_text", "string"),
                gbnfField("percentage", "string"))).append("\n");

        g.append("vessel ::= ").append(gbnfEnum(vessels)).append("\n");
        g.append("laterality ::= ").append(gbnfEnum(laterality)).append("\n");
        g.append("vessel-location ::= ").append(gbnfEnum(vesselLoc)).append("\n");
        g.append("velocity-type ::= ").append(gbnfEnum(velocityType)).append("\n");
        g.append("flow-direction ::= ").append(gbnfEnum(flowDirection)).append("\n");
        g.append("units ::= ").append(gbnfEnum(unitsEnum)).append("\n");

        // Same JSON string and whitespace rules as llama.cpp's grammars/json.gbnf.
        g.append("string ::= \"\\\"\" ( [^\"\\\\\\x7F\\x00-\\x1F] | \"\\\\\" ")
         .append("( [\"\\\\bfnrt] | \"u\" [0-9a-fA-F]{4} ) )* \"\\\"\"\n");
        g.append("ws ::= | \" \" | \"\\n\" [ \\t]{0,20}\n");
        return g.toString();
    }

    // A GBNF literal matching the given text exactly.
    private static String gbnfLit(String text) {
        return "\"" + text.replace("\\", "\\\\").replace("\"", "\\\"") + "\"";
    }

    // Alternatives matching a JSON string whose value is one of the given values.
    private static String gbnfEnum(List<String> values) {
        List<String> alts = new ArrayList<>();
        for (String v : values) {
            alts.add(gbnfLit("\"" + v + "\""));
        }
        return String.join(" | ", alts);
    }

    // "name": <valueRule>
    private static String gbnfField(String name, String valueRule) {
        return gbnfLit("\"" + name + "\"") + " ws " + gbnfLit(":") + " ws " + valueRule;
    }

    private static String gbnfObject(String... fields) {
        return gbnfLit("{") + " ws " + String.join(" ws " + gbnfLit(",") + " ws ", fields)
                + " ws " + gbnfLit("}");
    }

    private static String gbnfArray(String itemRule) {
        return gbnfLit("[") + " ws ( " + itemRule + " ( ws " + gbnfLit(",") + " ws " + itemRule
                + " )* )? ws " + gbnfLit("]");
    }

    private String postJson(String url, Object body) throws Exception {
        String json = OBJECT_MAPPER.writeValueAsString(body);
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(url))
                .header("Content-Type", "application/json")
                .timeout(REQUEST_TIMEOUT)
                .POST(HttpRequest.BodyPublishers.ofString(json))
                .build();

        HttpResponse<String> response;
        try {
            response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
        } catch (HttpTimeoutException e) {
            throw new RuntimeException("No response from llama-server within the HTTP timeout ("
                    + "connect " + CONNECT_TIMEOUT.toSeconds() + " s, request "
                    + REQUEST_TIMEOUT.toMinutes() + " min): " + e.getMessage(), e);
        }

        if (response.statusCode() < 200 || response.statusCode() >= 300) {
            // Safety net: if the character estimate let a too-long prompt through,
            // the server rejects it; treat that like the guard does.
            if (response.statusCode() == 400
                    && (response.body().contains("exceed_context_size")
                            || response.body().contains("exceeds the available context size"))) {
                throw new ContextBudget.DocumentTooLongException(
                        "Server rejected the prompt as larger than the slot context: " + response.body());
            }
            throw new RuntimeException(
                    "llama.cpp server error: " + response.statusCode() + "\n" + response.body());
        }
        return response.body();
    }

    // Typed exception so the caller can distinguish truncation (needs a larger
    // token budget) from malformed-but-complete output (needs the repair prompt).
    private static class TruncationException extends RuntimeException {
        private static final long serialVersionUID = 1L;
        final boolean repetitionSuspected;
        final int reasoningChars;
        final int contentChars;

        TruncationException(String message, boolean repetitionSuspected, int reasoningChars,
                int contentChars) {
            super(message);
            this.repetitionSuspected = repetitionSuspected;
            this.reasoningChars = reasoningChars;
            this.contentChars = contentChars;
        }
    }

    // -------------------------------------------------------------------------
    // Repetition detection
    // -------------------------------------------------------------------------

    private static final Pattern SOURCE_TEXT_FIELD =
            Pattern.compile("\"source_text\"\\s*:\\s*\"((?:[^\"\\\\]|\\\\.)*)\"");

    // For TRUNCATED output (not valid JSON): true when the same source_text value
    // appears LOOP_REPEAT_THRESHOLD or more times. Excerpts are required to be
    // unique in the report, so a genuine output has at most a couple of entries per
    // excerpt (e.g. its PSV and EDV entries).
    private static boolean looksLikeRepetitionLoop(String partialJson) {
        Map<String, Integer> counts = new HashMap<>();
        Matcher m = SOURCE_TEXT_FIELD.matcher(partialJson);
        while (m.find()) {
            if (counts.merge(m.group(1), 1, Integer::sum) >= LOOP_REPEAT_THRESHOLD) {
                return true;
            }
        }
        return false;
    }

    // For COMPLETE output: the highest number of identical entries in one section.
    private static int maxIdenticalEntries(ModelVascularResult r) {
        int max = 0;
        Map<ModelIcaCcaRatio, Integer> ratios = new HashMap<>();
        for (ModelIcaCcaRatio x : r.ICA_CCA_ratio()) {
            max = Math.max(max, ratios.merge(x, 1, Integer::sum));
        }
        Map<ModelFlowVelocity, Integer> velocities = new HashMap<>();
        for (ModelFlowVelocity x : r.flow_velocities()) {
            max = Math.max(max, velocities.merge(x, 1, Integer::sum));
        }
        Map<ModelStenosis, Integer> stenoses = new HashMap<>();
        for (ModelStenosis x : r.stenosis()) {
            max = Math.max(max, stenoses.merge(x, 1, Integer::sum));
        }
        return max;
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
            if (!VALID_VESSEL_LOCATIONS.contains(v.vessel_location())) {
                System.err.println("Warning: unexpected vessel_location value: " + v.vessel_location());
            }
            if (!VALID_UNITS.contains(v.units())) {
                System.err.println("Warning: unexpected units value: " + v.units());
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

        // Identical entries are kept (never collapsed), but many of them is a
        // strong sign of a repetition loop, so report it.
        int repeats = maxIdenticalEntries(result);
        if (repeats >= LOOP_REPEAT_THRESHOLD) {
            note("REPEATED_ENTRIES(" + repeats + ")");
            System.err.println("Warning: " + repeats + " identical entries in one section; "
                    + "possible repetition loop (entries are kept).");
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
                        v.velocity_type(), v.flow_direction(), v.source_text(), v.value(), v.units(),
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
                        + v.velocity_type() + " " + v.flow_direction() + ": " + v.value()
                        + " " + v.units()));

        model.stenosis().forEach(s ->
                System.out.println("  Stenosis " + s.laterality() + " " + s.vessel()
                        + ": " + s.percentage()));

        System.out.println("--------------------------------------------\n");
    }
}
