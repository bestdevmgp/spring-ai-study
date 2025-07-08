package com.dev.springaistudy.domain.openai.service;

import com.dev.springaistudy.domain.openai.entity.Chat;
import com.dev.springaistudy.domain.openai.repository.ChatRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;

@Service
@RequiredArgsConstructor
public class ChatService {
    private final ChatRepository repository;

    @Transactional(readOnly = true)
    public List<Chat> readAllChats(String userId) {
        return repository.findByUserIdOrderByCreatedAtAsc(userId);
    }
}
