package com.dev.springaistudy.api;

import com.dev.springaistudy.domain.openai.service.OpenAIService;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.ResponseBody;
import reactor.core.publisher.Flux;

import java.util.Map;

@Controller
@RequiredArgsConstructor
public class ChatController {
    private final OpenAIService openAIService;

    @ResponseBody
    @PostMapping("/chat")
    public String chat(@RequestBody Map<String, String> body) {
        return openAIService.generate(body.get("text"));
    }

    @ResponseBody
    @PostMapping("/chat/stream")
    public Flux<String> streamChat(@RequestBody Map<String, String> body) {
        return openAIService.generateStream(body.get("text"));
    }

    @GetMapping("/")
    public String chatPage() {
        return "chat";
    }
}
