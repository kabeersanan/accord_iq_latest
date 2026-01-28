import re
from typing import List

def recursive_chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """
    Splits text recursively for Legal RAG. 
    Respects strict chunk_size limits by drilling down separators.
    Preserves context via overlap.
    """
    # Base case: If text fits, return it as a single chunk
    if len(text) <= chunk_size:
        return [text]
    
    # Priority separators for Legal Docs
    # 1. Double Newline (Paragraphs/Articles)
    # 2. Newline (List items)
    # 3. Regex for Sentence ending (avoiding 'v.', 'No.', etc.)
    # 4. Space (Words)
    separators = [
        r"\n\n", 
        r"\n", 
        r"(?<!\bv)(?<!\bNo)(?<!\bVol)(?<!\bSec)\. ", # Negative lookbehind to protect legal abbr
        " ", 
        ""
    ]
    
    for separator in separators:
        # 1. CREATE SPLITS
        if separator == "":
            # Fallback: Hard character cut
            return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size-overlap)]
        
        # Use regex split if it's a regex pattern, otherwise normal split
        if separator in [r"\n\n", r"\n", " "]:
             splits = text.split(separator)
             separator_str = separator # What we re-insert
        else:
             # Complex regex split (keeps the delimiter usually, but here we simplify)
             splits = re.split(separator, text)
             separator_str = ". " 

        # If this separator didn't do anything (only 1 split), try the next one
        if len(splits) == 1:
            continue
            
        # 2. MERGE SPLITS INTO CHUNKS (With Accumulation)
        final_chunks = []
        current_chunk = []
        current_length = 0
        
        for split in splits:
            if not split.strip(): continue # Skip empty strings
            
            # If adding this split exceeds chunk_size...
            if current_length + len(split) + len(separator_str) > chunk_size:
                # A. Commit the current accumulated chunk
                if current_chunk:
                    joined_doc = separator_str.join(current_chunk)
                    final_chunks.append(joined_doc)
                    
                    # B. Handle Overlap (Backtrack logic)
                    # Keep the last few items of current_chunk to start the NEW chunk
                    # This is a simple sliding window overlap
                    overlap_len = 0
                    new_chunk = []
                    for item in reversed(current_chunk):
                        if overlap_len + len(item) < overlap:
                            new_chunk.insert(0, item)
                            overlap_len += len(item)
                        else:
                            break
                    current_chunk = new_chunk
                    current_length = sum(len(c) + len(separator_str) for c in current_chunk)

            # Add current split to accumulator
            current_chunk.append(split)
            current_length += len(split) + len(separator_str)
            
        # Append any leftovers
        if current_chunk:
            final_chunks.append(separator_str.join(current_chunk))
            
        # 3. RECURSIVE CHECK (The Critical Fix)
        # Are any of the resulting chunks STILL too big? 
        # If yes, we must recurse on THAT specific chunk using the *next* logic.
        # Note: A robust implementation usually passes the *index* of the separator to recurse efficiently.
        # But for simplicity, we verify validity:
        
        validated_chunks = []
        for chunk in final_chunks:
            if len(chunk) > chunk_size:
                # RECURSE: Call function again on this specific oversized chunk
                # Note: This naively restarts separator list. Optimized versions pass separator index.
                sub_chunks = recursive_chunk_text(chunk, chunk_size, overlap)
                validated_chunks.extend(sub_chunks)
            else:
                validated_chunks.append(chunk)
                
        return validated_chunks

    return [text]