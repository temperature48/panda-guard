
import yaml
import typer
import logging
import uvicorn
from typing import Optional, List, Dict, Any
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
import asyncio
import json

from panda_guard.pipelines.inference import InferPipeline, InferPipelineConfig
from panda_guard.utils import parse_configs_from_dict

app = typer.Typer(help="Deploy a model as a service")


# FastAPI models
class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]
    stream: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


class ChatResponse(BaseModel):
    messages: List[Dict[str, str]]
    usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def create_api(pipe, config_dict):
    """Create a FastAPI application with the given pipeline"""
    fast_api = FastAPI(
        title="Panda Guard API",
        description="API for interacting with language models through Panda Guard",
        version="0.1.0",
    )

    @fast_api.post("/v1/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest):
        try:
            # Convert messages from pydantic to dict
            messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]

            # Apply custom temperature if provided
            if request.temperature is not None:
                pipe.defender.target_llm_gen_config.temperature = request.temperature

            # Apply custom max_tokens if provided
            if request.max_tokens is not None:
                pipe.defender.target_llm_gen_config.max_n_tokens = request.max_tokens

            if request.stream:
                return await process_streaming_chat(pipe, messages)

            result = pipe(messages)

            usage = {
                "prompt_tokens": getattr(pipe.defender.target_llm, "prompt_tokens", 0),
                "completion_tokens": getattr(pipe.defender.target_llm, "completion_tokens", 0),
                "total_tokens": getattr(pipe.defender.target_llm, "total_tokens", 0)
            }

            return ChatResponse(
                messages=result["messages"],
                usage=usage
            )

        except Exception as e:
            logging.exception("Error in chat endpoint")
            raise HTTPException(status_code=500, detail=str(e))

    async def process_streaming_chat(pipe, messages):
        """处理流式聊天请求"""
        # 设置流式标志
        pipe.defender.target_llm_gen_config.stream = True

        # 处理消息并获取流式结果
        try:
            result = pipe(messages)

            # 更新检查流式生成器的逻辑
            # 检查结果的格式并调试记录
            logging.debug(f"Stream result type: {type(result)}")
            if isinstance(result, dict):
                logging.debug(f"Result keys: {result.keys()}")
                if "messages" in result:
                    logging.debug(f"Messages type: {type(result['messages'])}")

            # 创建一个异步生成器函数来处理流式输出
            async def stream_generator():
                try:
                    # 发送初始 SSE 字段
                    yield "data: " + json.dumps({"type": "start"}) + "\n\n"

                    # 根据返回结果的不同情况进行处理
                    if isinstance(result, dict) and "messages" in result:
                        # 可能是直接返回的生成器
                        if hasattr(result["messages"], "__iter__") and hasattr(result["messages"], "__next__"):
                            stream_content = result["messages"]
                        # 或者是列表中最后一个消息的内容是生成器
                        elif isinstance(result["messages"], list) and len(result["messages"]) > 0:
                            last_msg = result["messages"][-1]
                            if isinstance(last_msg, dict) and "content" in last_msg:
                                if hasattr(last_msg["content"], "__iter__") and hasattr(last_msg["content"],
                                                                                        "__next__"):
                                    stream_content = last_msg["content"]
                                else:
                                    # 不是流式，返回完整内容
                                    yield "data: " + json.dumps({
                                        "type": "chunk",
                                        "content": last_msg["content"]
                                    }) + "\n\n"
                                    yield "data: " + json.dumps({
                                        "type": "end",
                                        "content": last_msg["content"],
                                        "usage": {
                                            "prompt_tokens": getattr(pipe.defender.target_llm, "prompt_tokens", 0),
                                            "completion_tokens": getattr(pipe.defender.target_llm, "completion_tokens",
                                                                         0),
                                            "total_tokens": getattr(pipe.defender.target_llm, "total_tokens", 0)
                                        }
                                    }) + "\n\n"
                                    return
                            else:
                                raise ValueError("Invalid message format")
                        else:
                            raise ValueError("Result does not contain a valid stream")
                    else:
                        raise ValueError("Invalid result format")

                    # 累积完整响应用于计算tokens
                    full_content = ""

                    # 处理每个流式块
                    for chunk in stream_content:
                        full_content += chunk
                        # 发送 SSE 格式的数据
                        yield "data: " + json.dumps({
                            "type": "chunk",
                            "content": chunk
                        }) + "\n\n"

                        # 添加小延迟以避免客户端过载（可选）
                        await asyncio.sleep(0.01)

                    # 获取令牌使用情况
                    usage = {
                        "prompt_tokens": getattr(pipe.defender.target_llm, "prompt_tokens", 0),
                        "completion_tokens": getattr(pipe.defender.target_llm, "completion_tokens", 0),
                        "total_tokens": getattr(pipe.defender.target_llm, "total_tokens", 0)
                    }

                    # 发送完成事件，包括完整内容和使用情况
                    yield "data: " + json.dumps({
                        "type": "end",
                        "content": full_content,
                        "usage": usage
                    }) + "\n\n"

                except Exception as e:
                    # 处理流式过程中的错误
                    logging.exception("Error in streaming response")
                    yield "data: " + json.dumps({
                        "type": "error",
                        "error": str(e)
                    }) + "\n\n"

            # 返回流式响应
            return StreamingResponse(
                stream_generator(),
                media_type="text/event-stream"
            )

        except Exception as e:
            logging.exception("Error setting up streaming")
            raise HTTPException(status_code=500, detail=str(e))

    @fast_api.get("/health")
    async def health_check():
        return {"status": "ok", "model": config_dict["defender"]["target_llm_config"]["model_name"]}

    return fast_api


@app.command()
def start(
        config: Path = typer.Argument(..., help="Path to YAML configuration file"),
        defense: Optional[Path] = typer.Option(None, "--defense", "-d", help="Path to defense configuration file"),
        model: Optional[Path] = typer.Option(None, "--model", "-m", help="Path to model configuration file"),
        host: str = typer.Option("127.0.0.1", "--host", "-h", help="Host to bind the server to"),
        port: int = typer.Option(8000, "--port", "-p", help="Port to bind the server to"),
        device: Optional[str] = typer.Option(None, "--device", help="Device to run the model on (e.g., 'cuda:0')"),
        log_level: str = typer.Option("INFO", "--log-level", help="Logging level (DEBUG, INFO, WARNING, ERROR)"),
):
    """
    Start a service with the specified model using configuration from a YAML file.
    """
    # Set up logging
    logging.basicConfig(level=getattr(logging, log_level), format='%(asctime)s - %(levelname)s - %(message)s')

    try:
        if not config.exists():
            typer.echo(f"Error: Config file {config} not found", err=True)
            raise typer.Exit(1)

        config_dict = load_yaml(config)

        # Load and merge defense config if provided
        if defense:
            if not defense.exists():
                typer.echo(f"Error: Defense config file {defense} not found", err=True)
                raise typer.Exit(1)
            config_dict["defender"] = load_yaml(defense)

        # Load and merge model config if provided
        if model:
            if not model.exists():
                typer.echo(f"Error: Model config file {model} not found", err=True)
                raise typer.Exit(1)
            model_dict = load_yaml(model)
            config_dict["defender"]["target_llm_config"] = model_dict

        # Override device if specified
        if device:
            config_dict["defender"]["target_llm_config"]["device_map"] = device

        # Initialize the pipeline
        attacker_config, defender_config, _ = parse_configs_from_dict(config_dict)
        pipe = InferPipeline(
            InferPipelineConfig(
                attacker_config=attacker_config,
                defender_config=defender_config,
            ),
            verbose=False
        )

        # Create the FastAPI app
        fast_api = create_api(pipe, config_dict)

        # Print server info
        typer.echo(f"Starting server with model: {config_dict['defender']['target_llm_config']['model_name']}")
        typer.echo(f"Server will be available at http://{host}:{port}")
        typer.echo("API endpoints:")
        typer.echo("  - POST /v1/chat - Chat with the model")
        typer.echo("  - GET /health - Check server health")

        # Start the server
        uvicorn.run(fast_api, host=host, port=port)

    except Exception as e:
        typer.echo(f"Error starting server: {str(e)}", err=True)
        logging.exception("Exception starting server")
        raise typer.Exit(1)


def load_yaml(yaml_file):
    """Load YAML configuration file"""
    with open(yaml_file, 'r') as file:
        return yaml.safe_load(file)


if __name__ == "__main__":
    app()