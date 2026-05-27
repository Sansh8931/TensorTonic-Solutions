def text_chunking(tokens, chunk_size, overlap):
    """
    Split tokens into fixed-size chunks with optional overlap.
    """
    # Write code here
    m=chunk_size-overlap
    chunks=[]

    for i in range(0,len(tokens),chunk_size-overlap):
        chunk=tokens[i:i+chunk_size]

        if not chunk:
            break
        chunks.append(chunk)
        
        if i+chunk_size>=len(tokens) :
            break
    return chunks
            
        
    