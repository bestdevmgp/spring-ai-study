package com.dev.springaistudy.domain.openai.service;

import lombok.RequiredArgsConstructor;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.SystemMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.openai.*;
import org.springframework.stereotype.Service;
import reactor.core.publisher.Flux;

import java.util.List;

@Service
@RequiredArgsConstructor
public class OpenAIService {

    private final OpenAiChatModel chatModel;
    private final OpenAiEmbeddingModel embeddingModel;
    private final OpenAiImageModel imageModel;
    private final OpenAiAudioSpeechModel audioSpeechModel;
    private final OpenAiAudioTranscriptionModel audioTranscriptionModel;
    private final OpenAiChatModel openAiChatModel;

    // 1. chatModel : response
    public String generate(String text) {
        // 메시지
        SystemMessage systemMessage = new SystemMessage("");
        UserMessage userMessage = new UserMessage(text);
        AssistantMessage assistantMessage = new AssistantMessage("");

        OpenAiChatOptions options = OpenAiChatOptions.builder()
                .model("gpt-4.1-mini")
                .temperature(0.7)
                .build();

        // 프롬프트
        Prompt prompt = new Prompt(List.of(systemMessage, userMessage, assistantMessage), options);

        // 요청 및 응답
        ChatResponse response = openAiChatModel.call(prompt);
        return response.getResult().getOutput().getText();
    }

    // 2. chatModel : 실시간 response(stream)
    // Flux 자료형: 비동기 처리
    public Flux<String> generateStream(String text) {
        // 메시지
        SystemMessage systemMessage = new SystemMessage("");
        UserMessage userMessage = new UserMessage(text);
        AssistantMessage assistantMessage = new AssistantMessage("");

        // 옵션
        OpenAiChatOptions options = OpenAiChatOptions.builder()
                .model("gpt-4.1-mini")
                .temperature(0.7)
                .build();

        // 프롬프트
        Prompt prompt = new Prompt(List.of(systemMessage, userMessage, assistantMessage), options);

        // 요청 및 응답
        return openAiChatModel.stream(prompt)
                .mapNotNull(response -> response.getResult().getOutput().getText());
    }
}
