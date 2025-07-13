import re

class ContextWindow:
    def __init__(self, size=8192):
        self.size = size
        self.tokens = []
        # Define fluff patterns to remove at source
        self.fluff_patterns = [
            r'\b(um|uh|like|you know|basically|actually|literally|obviously|clearly|definitely|probably|maybe|perhaps|kinda|sorta)\b',
            r'\b(well|so|anyway|alright|ok|okay|right|yeah|yes|no|sure)\b',
            r'\s+',  # Multiple whitespace
            r'\n{3,}',  # Multiple newlines
            r'\.{2,}',  # Multiple dots
            r'\?{2,}',  # Multiple question marks
            r'!{2,}',  # Multiple exclamation marks
        ]

    def create_window(self, input_text):
        """Create context windows by splitting text into semantic chunks"""
        if isinstance(input_text, str):
            # For large texts, create semantic windows based on paragraphs and size
            # This replaces the complex multi-round partitioning
            windows = self._create_semantic_windows(input_text)
            
            # Ensure we have at least one window
            if not windows:
                windows = [input_text]
                
            self.tokens = windows
            return windows
        else:
            # Handle list input
            self.tokens = input_text if isinstance(input_text, list) else [str(input_text)]
            return self.tokens

    def _create_semantic_windows(self, text):
        """Create semantic windows that respect paragraph boundaries with fluff removal"""
        # Split by paragraphs first (double newlines)
        paragraphs = text.split('\n\n')
        windows = []
        current_window = ""
        
        for paragraph in paragraphs:
            # REMOVE FLUFF: Apply fluff removal to each paragraph before processing
            cleaned_paragraph = self._remove_fluff(paragraph)
            
            # Skip if paragraph becomes empty after fluff removal
            if not cleaned_paragraph.strip():
                continue
                
            # Check if adding this paragraph would exceed the size
            paragraph_words = len(cleaned_paragraph.split())
            current_words = len(current_window.split())
            
            if current_words + paragraph_words > self.size and current_window:
                # Current window is full, apply final fluff removal and start a new one
                cleaned_window = self._remove_fluff(current_window.strip())
                if cleaned_window.strip():  # Only add non-empty windows
                    windows.append(cleaned_window)
                current_window = cleaned_paragraph
            else:
                # Add paragraph to current window
                if current_window:
                    current_window += '\n\n' + cleaned_paragraph
                else:
                    current_window = cleaned_paragraph
        
        # Add the last window if it has content
        if current_window.strip():
            cleaned_final = self._remove_fluff(current_window.strip())
            if cleaned_final.strip():  # Only add non-empty windows
                windows.append(cleaned_final)
        
        # If no paragraph breaks, fall back to word-based chunking
        if not windows or len(windows) == 1 and len(text.split()) > self.size * 1.5:
            windows = self._fallback_word_chunking(text)
        
        return windows
    
    def _fallback_word_chunking(self, text):
        """Fallback to word-based chunking for texts without clear paragraph structure"""
        # Apply fluff removal first
        cleaned_text = self._remove_fluff(text)
        words = cleaned_text.split()
        windows = []
        
        # Create overlapping windows to preserve context
        overlap_size = self.size // 8  # 12.5% overlap
        
        for i in range(0, len(words), self.size - overlap_size):
            window_words = words[i:i + self.size]
            if window_words:
                window_text = ' '.join(window_words)
                # Apply final fluff removal to the window
                final_window = self._remove_fluff(window_text)
                if final_window.strip():  # Only add non-empty windows
                    windows.append(final_window)
        
        return windows

    def _remove_fluff(self, text):
        """Remove redundant content, filler words, and repetitive patterns at source"""
        if not text or not text.strip():
            return ""
        
        # Remove filler words and patterns
        cleaned = text
        for pattern in self.fluff_patterns:
            if pattern == r'\s+':
                # Replace multiple whitespace with single space
                cleaned = re.sub(pattern, ' ', cleaned, flags=re.IGNORECASE)
            elif pattern == r'\n{3,}':
                # Replace multiple newlines with double newline
                cleaned = re.sub(pattern, '\n\n', cleaned)
            elif pattern in [r'\.{2,}', r'\?{2,}', r'!{2,}']:
                # Replace multiple punctuation with single
                cleaned = re.sub(pattern, pattern[-3], cleaned)
            else:
                # Remove filler words (but preserve in code blocks)
                if not self._is_in_code_block(text, cleaned):
                    cleaned = re.sub(pattern, ' ', cleaned, flags=re.IGNORECASE)
        
        # Remove duplicate sentences (aggressive deduplication at source)
        cleaned = self._remove_duplicate_sentences(cleaned)
        
        # Remove repetitive phrases
        cleaned = self._remove_repetitive_phrases(cleaned)
        
        # Clean up extra spaces
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned
    
    def _is_in_code_block(self, original_text, current_text):
        """Check if we're inside a code block to preserve technical content"""
        # Simple heuristic: if text contains code markers, be conservative
        code_markers = ['```', '`', 'def ', 'class ', 'import ', 'from ', '#!/', '</', '/>', '{}', '[]', '()']
        return any(marker in original_text for marker in code_markers)
    
    def _remove_duplicate_sentences(self, text):
        """Remove duplicate sentences within the text"""
        sentences = re.split(r'[.!?]+', text)
        seen_sentences = set()
        unique_sentences = []
        
        for i, sentence in enumerate(sentences):
            sentence_clean = sentence.strip().lower()
            if sentence_clean and len(sentence_clean) > 10:  # Only check substantial sentences
                if sentence_clean not in seen_sentences:
                    seen_sentences.add(sentence_clean)
                    # Add back the original case sentence
                    unique_sentences.append(sentences[i].strip())
            elif sentence.strip():  # Keep short sentences as-is
                unique_sentences.append(sentences[i].strip())
        
        return '. '.join(s for s in unique_sentences if s).strip()
    
    def _remove_repetitive_phrases(self, text):
        """Remove phrases that repeat within a short span"""
        words = text.split()
        if len(words) < 20:
            return text  # Too short to have meaningful repetition
        
        # Look for phrase repetition in sliding windows
        cleaned_words = []
        i = 0
        while i < len(words):
            # Check for 3-5 word phrase repetition
            phrase_found = False
            for phrase_len in [5, 4, 3]:
                if i + phrase_len * 2 <= len(words):
                    phrase1 = ' '.join(words[i:i+phrase_len]).lower()
                    phrase2 = ' '.join(words[i+phrase_len:i+phrase_len*2]).lower()
                    
                    if phrase1 == phrase2 and len(phrase1) > 15:  # Substantial phrase
                        # Skip the repeated phrase
                        cleaned_words.extend(words[i:i+phrase_len])
                        i += phrase_len * 2
                        phrase_found = True
                        break
            
            if not phrase_found:
                cleaned_words.append(words[i])
                i += 1
        
        return ' '.join(cleaned_words)

    def get_tokens(self):
        return self.tokens