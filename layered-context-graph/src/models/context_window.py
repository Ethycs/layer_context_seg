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
                
            # Check if adding this paragraph would exceed the size (in characters)
            paragraph_chars = len(cleaned_paragraph)
            current_chars = len(current_window)
            
            if current_chars + paragraph_chars > self.size and current_window:
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
        # Check if single window is too large (more than 1.5x the character limit)
        if not windows or (len(windows) == 1 and len(windows[0]) > self.size * 1.5):
            windows = self._fallback_word_chunking(text)
        
        return windows
    
    def _fallback_word_chunking(self, text):
        """
        Fallback to word-based chunking for texts without clear paragraph structure
        Using percolation theory optimal overlap (15-30%) for knowledge connectivity
        """
        # Apply minimal formatting cleanup only
        cleaned_text = self._remove_fluff(text)
        words = cleaned_text.split()
        windows = []
        
        # Calculate optimal overlap based on percolation theory (15-30%)
        # Closer to 15% for long texts, closer to 30% for shorter texts
        # This creates a "phase transition" where information can percolate across the graph
        
        # Convert size from characters to approximate words (avg 5 chars per word)
        target_words_per_window = self.size // 5
        
        if len(words) > target_words_per_window * 3:
            # For very long texts, use lower overlap (15%)
            overlap_ratio = 0.15
        elif len(words) > target_words_per_window:
            # For medium texts, use middle overlap (20%)
            overlap_ratio = 0.20
        else:
            # For shorter texts, use higher overlap (25%)
            overlap_ratio = 0.25
            
        overlap_size = int(target_words_per_window * overlap_ratio)
        
        # Create overlapping windows to enable percolation
        for i in range(0, len(words), target_words_per_window - overlap_size):
            window_words = words[i:i + target_words_per_window]
            if window_words:
                window_text = ' '.join(window_words)
                # Only minimal formatting cleanup
                final_window = self._remove_fluff(window_text)
                if final_window.strip():  # Only add non-empty windows
                    windows.append(final_window)
        
        return windows

    def _remove_fluff(self, text):
        """Preserve all formatting - only remove truly redundant content"""
        if not text or not text.strip():
            return ""
        
        # Preserve all formatting - only clean up excessive blank lines
        cleaned = text
        
        # Only normalize excessive blank lines (3+ newlines to 2 newlines)
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        
        # Don't strip - preserve leading/trailing whitespace as it might be meaningful
        return cleaned
    
    def _is_in_code_block(self, original_text, current_text):
        """Check if we're inside a code block to preserve technical content"""
        # Simple heuristic: if text contains code markers, be conservative
        code_markers = ['```', '`', 'def ', 'class ', 'import ', 'from ', '#!/', '</', '/>', '{}', '[]', '()']
        return any(marker in original_text for marker in code_markers)
    

    def get_tokens(self):
        return self.tokens