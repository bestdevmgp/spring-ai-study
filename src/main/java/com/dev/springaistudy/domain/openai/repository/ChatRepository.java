package com.dev.springaistudy.domain.openai.repository;

import com.dev.springaistudy.domain.openai.entity.Chat;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;

public interface ChatRepository extends JpaRepository<Chat, Long> {

    List<Chat> findByUserIdOrderByCreatedAtAsc(String userId);
}
