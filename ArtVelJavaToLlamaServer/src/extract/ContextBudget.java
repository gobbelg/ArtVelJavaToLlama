package extract;

/**
 * Decides, BEFORE any model call, whether a document can be processed reliably in
 * one server slot, and how much output room it leaves.
 *
 * Everything for one request has to fit in a single slot:
 *
 *   system prompt + chat-template/wrapper overhead + report + generated output
 *       <= slot size
 *
 * The output cannot be predicted, so a fixed amount (outputReserveTokens, the
 * largest legitimate output expected) is reserved in every slot. A document is
 * admitted only if what remains after the system prompt, overhead, reserve and
 * safety margin is large enough to hold it:
 *
 *   maxDocumentTokens = slotSize - systemPrompt - OVERHEAD - reserve - MARGIN
 *
 * Documents longer than that are rejected with DocumentTooLongException so the
 * caller can set them aside for a run with fewer, larger slots.
 *
 * Token counts are ESTIMATED from character counts (CHARS_PER_TOKEN), so no
 * server round trip is needed. The ratio is deliberately pessimistic. The
 * extractor records real prompt_tokens/completion_tokens from the server and
 * reports the lowest characters-per-token actually observed, so the estimate
 * can be checked and tuned.
 */
public final class ContextBudget {

    /** Estimated characters per token. Lower = more conservative (more documents skipped). */
    public static final double CHARS_PER_TOKEN = 3.0;

    /** Chat-template tokens plus the fixed wrapper text around the report. */
    public static final int PROMPT_OVERHEAD_TOKENS = 128;

    /** Extra slack for estimation error. */
    public static final int SAFETY_MARGIN_TOKENS = 256;

    private static final int MIN_DOCUMENT_TOKENS = 256;

    /**
     * Thrown when a document cannot fit in a slot. Not a processing failure:
     * callers should collect these documents for a later pass with larger slots.
     */
    public static class DocumentTooLongException extends RuntimeException {
        private static final long serialVersionUID = 1L;

        public DocumentTooLongException(String message) {
            super(message);
        }
    }

    private final int slotSize;
    private final int systemPromptTokens;
    private final int outputReserveTokens;
    private final double charsPerToken;
    private final int promptOverheadTokens;
    private final int safetyMarginTokens;

    /** Uses the default estimate constants above. */
    public ContextBudget(int slotSize, String systemPrompt, int outputReserveTokens) {
        this(slotSize, systemPrompt, outputReserveTokens, CHARS_PER_TOKEN,
                PROMPT_OVERHEAD_TOKENS, SAFETY_MARGIN_TOKENS);
    }

    /** The estimate constants can be tuned per model (see ExtractionOptions). */
    public ContextBudget(int slotSize, String systemPrompt, int outputReserveTokens,
            double charsPerToken, int promptOverheadTokens, int safetyMarginTokens) {
        this.slotSize = slotSize;
        this.charsPerToken = charsPerToken;
        this.promptOverheadTokens = promptOverheadTokens;
        this.safetyMarginTokens = safetyMarginTokens;
        this.systemPromptTokens = estimateTokens(systemPrompt);
        this.outputReserveTokens = outputReserveTokens;

        if (maxDocumentTokens() < MIN_DOCUMENT_TOKENS) {
            throw new IllegalArgumentException("Slot size " + slotSize + " is too small: system prompt (~"
                    + systemPromptTokens + ") + overhead + output reserve (" + outputReserveTokens
                    + ") + margin leave only " + maxDocumentTokens() + " tokens for a report. "
                    + "Use fewer slots (larger slot size) or lower the output reserve.");
        }
    }

    public int estimateTokens(String text) {
        return (int) Math.ceil(text.length() / charsPerToken);
    }

    public double charsPerToken() {
        return charsPerToken;
    }

    public int slotSize() {
        return slotSize;
    }

    public int maxDocumentTokens() {
        return slotSize - systemPromptTokens - promptOverheadTokens
                - outputReserveTokens - safetyMarginTokens;
    }

    public int maxDocumentChars() {
        return (int) Math.floor(maxDocumentTokens() * charsPerToken);
    }

    /** Throws DocumentTooLongException if the document is above the limit. */
    public void requireFits(String document) {
        if (document.length() > maxDocumentChars()) {
            throw new DocumentTooLongException("Too long for the current slot size: "
                    + document.length() + " characters (about " + estimateTokens(document)
                    + " tokens) exceeds the limit of " + maxDocumentChars() + " characters "
                    + "(slot size " + slotSize + " tokens, output reserve " + outputReserveTokens + ").");
        }
    }

    /** Estimated tokens still free for output after the prompt (never negative). */
    public int availableOutputTokens(String document) {
        int prompt = systemPromptTokens + promptOverheadTokens + estimateTokens(document);
        return Math.max(0, slotSize - prompt - safetyMarginTokens);
    }

    public String describe() {
        return "Context budget: slot size " + slotSize + " tokens; system prompt ~" + systemPromptTokens
                + "; output reserve " + outputReserveTokens + "; largest admitted report ~"
                + maxDocumentTokens() + " tokens (" + maxDocumentChars() + " characters at "
                + charsPerToken + " chars/token). Longer reports are set aside for a larger-slot pass.";
    }
}
