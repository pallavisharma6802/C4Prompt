from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from app.compressor import compress, count_tokens


class CompressRequest(BaseModel):
    prompt: str
    use_tiktoken: bool = False
    aggressive: bool = False
    remove_stopwords: bool = False
    remove_punctuation: bool = False
    light_stemming: bool = False
    super_aggressive: bool = False
    encode: bool = False


app = FastAPI()

# Mount the `static` directory so the frontend can be served at /static/index.html
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.post("/compress")
async def compress_endpoint(req: CompressRequest):
    original_tokens = count_tokens(req.prompt, use_tiktoken=req.use_tiktoken)
    compressed_text, meta = compress(
        req.prompt,
        aggressive=req.aggressive,
        remove_stopwords_opt=req.remove_stopwords,
        remove_punctuation_opt=req.remove_punctuation,
        light_stemming_opt=req.light_stemming,
        encode=req.encode,
        use_tiktoken=req.use_tiktoken,
    )
    # If super_aggressive is requested, re-run with stronger heuristics
    if req.super_aggressive:
        compressed_text, meta = compress(
            req.prompt,
            aggressive=True,
            remove_stopwords_opt=True,
            remove_punctuation_opt=True,
            light_stemming_opt=True,
            use_tiktoken=req.use_tiktoken,
        )
    compressed_tokens = count_tokens(compressed_text, use_tiktoken=req.use_tiktoken)
    return {
        "original_prompt": req.prompt,
        "compressed_prompt": compressed_text,
        "original_tokens": original_tokens,
        "compressed_tokens": compressed_tokens,
        "tokens_saved": original_tokens - compressed_tokens,
        "meta": meta,
    }


@app.get("/")
async def root():
    # Redirect to the static frontend page
    return RedirectResponse(url="/static/index.html")
