import type { OpenAI } from "openai";
import type {
  ResponseCreateParams,
  Response,
} from "openai/resources/responses/responses";

import { log } from "./logger/log.js";
import { responsesCreateViaChatCompletions } from "./responses.js";

export interface ModelApiAdapter {
  /**
   * Creates a response using the appropriate API endpoint for the given model.
   * Automatically handles fallbacks for models that don't support certain APIs.
   */
  createResponse(
    openai: OpenAI,
    params: ResponseCreateParams,
    onFallbackMessage?: (message: string) => void,
  ): Promise<AsyncGenerator<unknown> | Response>;

  /**
   * Checks if a model requires special handling (e.g., search preview models)
   */
  requiresSpecialHandling(model: string): boolean;

  /**
   * Returns the appropriate provider for a model (e.g., "azure" for chat/completions fallback)
   */
  getEffectiveProvider(model: string, originalProvider?: string): string;
}

class DefaultModelApiAdapter implements ModelApiAdapter {
  private static readonly SEARCH_PREVIEW_MODELS = [
    "gpt-4o-search-preview",
    "gpt-4o-mini-search-preview",
    "gpt-4o-search-preview-2025-03-11",
    "gpt-4o-mini-search-preview-2025-03-11",
  ];

  requiresSpecialHandling(model: string): boolean {
    return this.isSearchPreviewModel(model);
  }

  getEffectiveProvider(model: string, originalProvider?: string): string {
    // If it's a search preview model, force Azure provider to use chat/completions
    if (this.isSearchPreviewModel(model)) {
      return "azure";
    }
    return originalProvider || "openai";
  }

  private isSearchPreviewModel(model: string): boolean {
    return DefaultModelApiAdapter.SEARCH_PREVIEW_MODELS.some((_searchModel) =>
      model.includes("search-preview"),
    );
  }

  async createResponse(
    openai: OpenAI,
    params: ResponseCreateParams,
    onFallbackMessage?: (message: string) => void,
  ): Promise<AsyncGenerator<unknown> | Response> {
    const isSearchPreview = this.isSearchPreviewModel(params.model);

    // For search preview models, go directly to chat/completions
    if (isSearchPreview) {
      log(
        `Model ${params.model} detected as search preview, using chat/completions directly`,
      );

      if (onFallbackMessage) {
        onFallbackMessage(`🔍 Using web search with ${params.model}...`);
      }

      // Remove tools for search preview models as they don't support function calling
      const chatParams = {
        ...params,
        tools: [], // Remove tools
        tool_choice: "none" as const, // Disable tool choice
      };

      return responsesCreateViaChatCompletions(
        openai,
        chatParams as ResponseCreateParams & { stream: true },
      );
    }

    // For regular models, use the standard responses API
    return openai.responses.create(params);
  }
}

// Singleton instance
const modelApiAdapter = new DefaultModelApiAdapter();

export { modelApiAdapter };
export default modelApiAdapter;
