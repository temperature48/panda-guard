# src/panda_guard/cli/chat.py
import os
import sys

import yaml
import typer
import logging
from typing import Optional, Iterator, Dict, Any
from pathlib import Path
from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live
from rich.text import Text

from panda_guard.pipelines.inference import InferPipeline, InferPipelineConfig
from panda_guard.utils import parse_configs_from_dict

app = typer.Typer(help="Interactive chat with language models")
console = Console()


def is_iterator(obj):
    """Check if an object is an iterator but not a list or other sequence."""
    return (
            hasattr(obj, "__iter__") and
            not isinstance(obj, (list, dict, str, bytes, tuple)) and
            hasattr(obj, "__next__")
    )


@app.command()
def start(
        config: Path = typer.Argument(..., help="Path to YAML configuration file"),
        defense: Optional[Path] = typer.Option(None, "--defense", "-d", help="Path to defense configuration file"),
        model: Optional[Path] = typer.Option(None, "--model", "-m", help="Path to model configuration file"),
        temperature: Optional[float] = typer.Option(None, "--temperature", "-t", help="Override temperature setting"),
        device: Optional[str] = typer.Option(None, "--device", help="Device to run the model on (e.g., 'cuda:0')"),
        log_level: str = typer.Option("WARNING", "--log-level", help="Logging level (DEBUG, INFO, WARNING, ERROR)"),
        output: Optional[Path] = typer.Option(None, "--output", "-o", help="Save chat history to file"),
        stream: bool = typer.Option(True, "--stream/--no-stream", help="Enable/disable streaming output"),
):
    """
    Start an interactive chat session using configuration from a YAML file.
    """
    # Set up logging
    logging.basicConfig(level=getattr(logging, log_level), format='%(asctime)s - %(levelname)s - %(message)s')

    # Load configuration
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
            if str(model).endswith(".yaml"):
                if not model.exists():
                    typer.echo(f"Error: Model config file {model} not found", err=True)
                    raise typer.Exit(1)
                model_dict = load_yaml(model)
                config_dict["defender"]["target_llm_config"] = model_dict
            else:
                config_dict["defender"]["target_llm_config"]['model_name'] = str(model)

        # Override device if specified
        if device:
            config_dict["defender"]["target_llm_config"]["device_map"] = device

        # Override temperature if specified
        if temperature is not None:
            config_dict["defender"]["target_llm_gen_config"]["temperature"] = temperature

        # Enable or disable streaming in the configuration
        # if "target_llm_gen_config" not in config_dict.get("defender", {}):
        #     config_dict["defender"]["target_llm_gen_config"] = {}
        config_dict["defender"]["target_llm_gen_config"]["stream"] = stream

        # Initialize the pipeline
        attacker_config, defender_config, _ = parse_configs_from_dict(config_dict)

        print(defender_config)

        pipe = InferPipeline(
            InferPipelineConfig(
                attacker_config=attacker_config,
                defender_config=defender_config,
            ),
            verbose=False
        )

        console.print(
            f"[bold green]Chat initialized with {config_dict['defender']['target_llm_config']['model_name']}[/bold green]")
        console.print("[bold]Type your message (or '/exit' to quit, '/reset' to start new conversation)[/bold]")

        # Initialize chat history
        messages = []

        # Chat loop
        while True:
            user_input = console.input("[bold blue]User:[/bold blue] ")

            if user_input.lower() == "/exit":
                break

            if user_input.lower() == "/reset":
                messages = []
                pipe.reset()
                console.print("[bold yellow]Conversation reset[/bold yellow]")
                continue

            # Create or update messages
            if not messages:
                messages = [{"role": "user", "content": user_input}]
            else:
                messages.append({"role": "user", "content": user_input})

            try:
                # Process through pipeline
                result = pipe(messages)

                # Check if the result is streaming or regular
                if stream and isinstance(result, dict) and "messages" in result and is_iterator(result["messages"]):
                    # Handle streaming response
                    console.print("[bold green]Assistant:[/bold green]")

                    full_response = ""

                    try:
                        generator = result["messages"]

                        for text_chunk in generator:
                            full_response += text_chunk
                            console.print(text_chunk, end="")
                            sys.stdout.flush()

                        console.print()

                    except Exception as e:
                        console.print(f"\n[bold red]Error during streaming: {str(e)}[/bold red]")
                        logging.exception("Exception during streaming")
                else:
                    # Handle regular non-streaming response
                    if isinstance(result, dict) and "messages" in result:
                        messages = result["messages"]
                    else:
                        # If result is not a dict with messages, it might be the messages directly
                        messages = result

                    # Display assistant response
                    assistant_response = messages[-1]["content"]
                    console.print("[bold green]Assistant:[/bold green]")
                    console.print(Markdown(assistant_response))

            except Exception as e:
                console.print(f"[bold red]Error: {str(e)}[/bold red]")
                logging.exception("Exception during chat")
                continue

        # Save chat history if requested
        if output:
            import json
            with open(output, 'w') as f:
                json.dump(messages, f, indent=2)
            console.print(f"[bold]Chat history saved to {output}[/bold]")

    except Exception as e:
        typer.echo(f"Error during chat: {str(e)}", err=True)
        logging.exception("Exception during chat")
        raise typer.Exit(1)


def load_yaml(yaml_file):
    """Load YAML configuration file"""
    with open(yaml_file, 'r') as file:
        return yaml.safe_load(file)


if __name__ == "__main__":
    app()