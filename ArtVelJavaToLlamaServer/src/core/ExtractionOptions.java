package core;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Everything about HOW the model is called that can change speed or accuracy and is
 * therefore an experiment variable rather than a fixed part of the code. Built from
 * the settings (AppSettings.extractionOptions()); defaults() reproduces the values
 * that used to be hard-coded in LlamaVascularExtractor and ContextBudget.
 *
 *   maxTokens / retryMaxTokens / repairMaxTokens   generation ceilings (primary call,
 *                                                  larger-budget retry, JSON repair)
 *   temperature, topK, topP, repeatPenalty, seed   sampling. The defaults are greedy
 *                                                  decoding (temperature 0, top_k 1).
 *                                                  seed &lt; 0 means "do not send a seed".
 *   disableThinking                                sends enable_thinking=false (Gemma 4
 *                                                  and other models with a thought channel)
 *   useGrammar                                     send a GBNF grammar instead of a
 *                                                  json_schema response_format
 *   charsPerToken, promptOverheadTokens,           the ContextBudget estimate; tune per
 *   safetyMarginTokens                             model, since tokenizers differ
 */
public final class ExtractionOptions {

    public final int maxTokens;
    public final int retryMaxTokens;
    public final int repairMaxTokens;
    public final double temperature;
    public final int topK;
    public final double topP;
    public final double repeatPenalty;
    public final int seed;
    public final boolean disableThinking;
    public final boolean useGrammar;
    public final double charsPerToken;
    public final int promptOverheadTokens;
    public final int safetyMarginTokens;

    public ExtractionOptions(int maxTokens, int retryMaxTokens, int repairMaxTokens,
            double temperature, int topK, double topP, double repeatPenalty, int seed,
            boolean disableThinking, boolean useGrammar,
            double charsPerToken, int promptOverheadTokens, int safetyMarginTokens) {
        this.maxTokens = maxTokens;
        this.retryMaxTokens = retryMaxTokens;
        this.repairMaxTokens = repairMaxTokens;
        this.temperature = temperature;
        this.topK = topK;
        this.topP = topP;
        this.repeatPenalty = repeatPenalty;
        this.seed = seed;
        this.disableThinking = disableThinking;
        this.useGrammar = useGrammar;
        this.charsPerToken = charsPerToken;
        this.promptOverheadTokens = promptOverheadTokens;
        this.safetyMarginTokens = safetyMarginTokens;
    }

    /** The values that were hard-coded before these became settings. */
    public static ExtractionOptions defaults() {
        return new ExtractionOptions(6000, 8192, 2048, 0.0, 1, 1.0, 1.0, -1, true, false,
                ContextBudget.CHARS_PER_TOKEN, ContextBudget.PROMPT_OVERHEAD_TOKENS,
                ContextBudget.SAFETY_MARGIN_TOKENS);
    }

    /** For the run manifest. */
    public Map<String, Object> asMap() {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("max_tokens", maxTokens);
        m.put("retry_max_tokens", retryMaxTokens);
        m.put("repair_max_tokens", repairMaxTokens);
        m.put("temperature", temperature);
        m.put("top_k", topK);
        m.put("top_p", topP);
        m.put("repeat_penalty", repeatPenalty);
        m.put("seed", seed < 0 ? "not sent" : Integer.toString(seed));
        m.put("disable_thinking", disableThinking);
        m.put("use_grammar_instead_of_schema", useGrammar);
        m.put("chars_per_token", charsPerToken);
        m.put("prompt_overhead_tokens", promptOverheadTokens);
        m.put("safety_margin_tokens", safetyMarginTokens);
        return m;
    }
}
