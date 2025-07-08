package com.dev.springaistudy.domain.openai.service;

import com.dev.springaistudy.domain.openai.entity.Chat;
import com.dev.springaistudy.domain.openai.repository.ChatRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.ai.audio.transcription.AudioTranscriptionPrompt;
import org.springframework.ai.audio.transcription.AudioTranscriptionResponse;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.memory.ChatMemory;
import org.springframework.ai.chat.memory.ChatMemoryRepository;
import org.springframework.ai.chat.memory.MessageWindowChatMemory;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.MessageType;
import org.springframework.ai.chat.messages.SystemMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.embedding.Embedding;
import org.springframework.ai.embedding.EmbeddingOptions;
import org.springframework.ai.embedding.EmbeddingRequest;
import org.springframework.ai.embedding.EmbeddingResponse;
import org.springframework.ai.image.ImagePrompt;
import org.springframework.ai.image.ImageResponse;
import org.springframework.ai.openai.*;
import org.springframework.ai.openai.api.OpenAiAudioApi;
import org.springframework.ai.openai.audio.speech.SpeechPrompt;
import org.springframework.ai.openai.audio.speech.SpeechResponse;
import org.springframework.core.io.Resource;
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
    private final ChatMemoryRepository chatMemoryRepository;
    private final ChatRepository chatRepository;
    private final OpenAiChatModel openAiChatModel;

    // 1-1. chatModel : response
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
        ChatResponse response = chatModel.call(prompt);
        return response.getResult().getOutput().getText();
    }

    // 1-2. chatModel : 실시간 response(stream)
    // Flux 자료형: 비동기 처리
    public Flux<String> generateStream(String text) {

        ChatClient chatClient = ChatClient.create(openAiChatModel);

        // 유저&페이지별 ChatMemory를 관리하기 위한 key (우선은 명시적으로)
        String userId = "xxxjjhhh" + "_" + "3";

        // 전체 대화 저장용
        Chat userMessage = new Chat();
        userMessage.setUserId(userId);
        userMessage.setType(MessageType.USER);
        userMessage.setContent(text);

        // 메시지
        ChatMemory chatMemory = MessageWindowChatMemory.builder()
                .maxMessages(10) // 10개 이전 메시지까지 참조
                .chatMemoryRepository(chatMemoryRepository)
                .build();
        chatMemory.add(userId, new UserMessage(text)); // 신규 메시지도 추가

        // 옵션
        OpenAiChatOptions options = OpenAiChatOptions.builder()
                .model("gpt-4.1-mini")
                .temperature(0.7)
                .build();

        // 프롬프트
        Prompt prompt = new Prompt(chatMemory.get(userId), options);

        // 논블로킹으로 오는 응답 메시지를 저장할 임시 버퍼
        StringBuilder responseBuffer = new StringBuilder();

        // 요청 및 응답
        return chatModel.stream(prompt)
                .mapNotNull(response -> {
                    String token = response.getResult().getOutput().getText();
                    responseBuffer.append(token);
                    return token;
                })
                .doOnComplete(() -> {

                    chatMemory.add(userId, new AssistantMessage(responseBuffer.toString()));
                    chatMemoryRepository.saveAll(userId, chatMemory.get(userId));

                    // 전체 대화 저장용
                    Chat assistantMessage = new Chat();
                    assistantMessage.setUserId(userId);
                    assistantMessage.setType(MessageType.ASSISTANT);
                    assistantMessage.setContent(responseBuffer.toString());
                    chatRepository.saveAll(List.of(userMessage, assistantMessage));
                });
    }

    // 2. 임베딩 api 호출
    public List<float[]> generateEmbedding(List<String> texts, String model) {
        // 옵션
        EmbeddingOptions embeddingOptions = OpenAiEmbeddingOptions.builder()
                .model(model).build();

        // 프롬프트
        EmbeddingRequest prompt = new EmbeddingRequest(texts, embeddingOptions);

        // 요청 및 운동
        EmbeddingResponse response = embeddingModel.call(prompt);
        return response.getResults().stream()
                .map(Embedding::getOutput)
                .toList();
    }

    // 3. image 모델 : DALL-E
    public List<String> generateImages(String text, int count, int height, int width) {

        // 옵션
        OpenAiImageOptions imageOptions = OpenAiImageOptions.builder()
                .quality("hd")
                .N(count)
                .height(height)
                .width(width)
                .build();

        // 프롬프트
        ImagePrompt prompt = new ImagePrompt(text, imageOptions);

        // 요청 및 응답
        ImageResponse response = imageModel.call(prompt);
        return response.getResults().stream()
                .map(image -> image.getOutput().getUrl())
                .toList();
    }

    // 4. TTS
    public byte[] tts(String text) {

        // 옵션
        OpenAiAudioSpeechOptions speechOptions = OpenAiAudioSpeechOptions.builder()
                .responseFormat(OpenAiAudioApi.SpeechRequest.AudioResponseFormat.MP3)
                .speed(1.0f)
                .model(OpenAiAudioApi.TtsModel.TTS_1.value)
                .build();

        // 프롬프트
        SpeechPrompt prompt = new SpeechPrompt(text, speechOptions);

        // 요청 및 응답
        SpeechResponse response = audioSpeechModel.call(prompt);
        return response.getResult().getOutput();
    }

    // 5. STT
    public String stt(Resource audioFile) {

        // 옵션
        OpenAiAudioApi.TranscriptResponseFormat responseFormat = OpenAiAudioApi.TranscriptResponseFormat.VTT;
        OpenAiAudioTranscriptionOptions transcriptionOptions = OpenAiAudioTranscriptionOptions.builder()
                .language("ko") // 인식할 언어
                .prompt("Ask not this, but ask that") // 음성 인식 전 참고할 텍스트 프롬프트
                .temperature(0f)
                .model(OpenAiAudioApi.TtsModel.TTS_1.value)
                .responseFormat(responseFormat) // 결과 타입 지정 VTT 자막형식
                .build();

        // 프롬프트
        AudioTranscriptionPrompt prompt = new AudioTranscriptionPrompt(audioFile, transcriptionOptions);

        // 요청 및 응답
        AudioTranscriptionResponse response = audioTranscriptionModel.call(prompt);
        return response.getResult().getOutput();
    }
}
