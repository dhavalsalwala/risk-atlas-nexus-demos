import asyncio

#!/usr/bin/env python
import os

import streamlit as st
from acp_sdk.client import Client
from acp_sdk.models import Message, MessagePart

from gaf_guard.toolkit.file_utils import resolve_file_paths


os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import asyncio
import json
import os
import signal
from typing import Annotated, Dict, List

import typer

from gaf_guard.core.models import WorkflowStepMessage
from gaf_guard.toolkit.enums import MessageType, Role


# def signal_handler(sig, frame):
#     print("Exiting...")
#     for task in asyncio.tasks.all_tasks():
#         task.cancel()
#     sys.exit(0)


def pprint(key, value):
    if isinstance(value, List) or isinstance(value, Dict):
        return json.dumps(value, indent=2)
    elif isinstance(value, str) and key.endswith("alert"):
        return f"[red]{value}[/red]"
    else:
        return value


# signal.signal(signal.SIGINT, signal_handler)

app = typer.Typer()

run_configs = {
    "RiskGeneratorAgent": {
        "risk_questionnaire_cot": "examples/data/chain_of_thought/risk_questionnaire.json",
        "risk_generation_cot": "examples/data/chain_of_thought/risk_generation.json",
    },
    "DriftMonitoringAgent": {
        "drift_threshold": 8,
        "drift_monitoring_cot": "examples/data/chain_of_thought/drift_monitoring.json",
    },
}
resolve_file_paths(run_configs)


async def run_app(host, port):

    if "client_session" not in st.session_state:
        client = Client(base_url=f"http://{host}:{port}")
        st.session_state.client_session = client.session()
        st.session_state.input_message_type = MessageType.WORKFLOW_INPUT
        st.session_state.input_message_query = "Enter user intent here"
        st.session_state.input_message_key = "user_intent"
        st.session_state.messages = [{"role": "System", "content": "New Session 👇"}]

    st.set_page_config(
        page_title="GAF Guard - A real-time monitoring system for risk assessment and drift monitoring.",
        layout="wide",  # This sets the app to wide mode
        # initial_sidebar_state="expanded",
    )

    st.title(
        ":yellow[GAF Guard]",
        text_alignment="center",
    )
    st.subheader(
        "A real-time monitoring system for risk assessment and drift monitoring",
        text_alignment="center",
    )

    # st.caption(
    #     f"{datetime.now().strftime('%d-%m-%Y %H:%M:%S')}   🚀   :small[Connected to :yellow[GAF Guard] Server at]",
    #     text_alignment="center",
    # )

    st.markdown(
        f":violet-badge[:material/rocket_launch: Connected to :yellow[GAF Guard] Server:] :orange-badge[:material/check: {host}:{port}]",
        text_alignment="center",
    )

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    async with st.session_state.client_session:
        # Accept user input
        if input_message_response := st.chat_input(
            st.session_state["input_message_query"]
        ):
            # Add user message to chat history
            # st.session_state.messages.append({"role": "user", "content": user_intent})
            # Display user message in chat message container
            # with st.chat_message("user"):
            #     st.markdown("User Intent: " + user_intent)

            # chat_container.empty()

            # Display assistant response in chat message container
            # with st.chat_message("assistant"):
            #     message_placeholder = st.empty()
            #     full_response = ""
            #     assistant_response = random.choice(
            #         [
            #             "Hello there! How can I assist you today?",
            #             "Hi, human! Is there anything I can help you with?",
            #             "Do you need help?",
            #         ]
            #     )
            #     # Simulate stream of response with milliseconds delay
            #     for chunk in assistant_response.split():
            #         full_response += chunk + " "
            #         time.sleep(0.05)
            #         # Add a blinking cursor to simulate typing
            #         message_placeholder.markdown(full_response + "▌")
            #     message_placeholder.markdown(full_response)
            # # Add assistant response to chat history
            # st.session_state.messages.append(
            #     {"role": "assistant", "content": full_response}
            # )

            COMPLETED = False
            while True:
                async for event in st.session_state.client_session.run_stream(
                    agent="orchestrator",
                    input=[
                        Message(
                            parts=[
                                MessagePart(
                                    content=WorkflowStepMessage(
                                        step_name="GAF Guard Client",
                                        step_type=st.session_state[
                                            "input_message_type"
                                        ],
                                        step_role=Role.USER,
                                        content={
                                            st.session_state[
                                                "input_message_key"
                                            ]: input_message_response
                                        },
                                        run_configs=run_configs,
                                    ).model_dump_json(),
                                    content_type="text/plain",
                                )
                            ]
                        )
                    ],
                ):
                    if event.type == "message.part":
                        message = WorkflowStepMessage(**json.loads(event.part.content))

                        if message.step_type == MessageType.WORKFLOW_STARTED:
                            with st.chat_message(message.step_role.value):
                                st.markdown(
                                    f":blue[{message.step_name}]: {message.content}"
                                )
                        elif message.step_type == MessageType.STEP_STARTED:
                            # if message.step_desc:
                            #     console.print(message.step_desc, **message.step_kwargs)
                            with st.chat_message(message.step_role.value):
                                st.markdown(
                                    f"\n:blue[Workflow Step:] {message.step_name}....Started"
                                )
                                st.session_state.messages.append(
                                    {
                                        "role": message.step_role.value,
                                        "content": f"\n:blue[Workflow Step:] {message.step_name}....Started",
                                    }
                                )
                        elif message.step_type == MessageType.STEP_COMPLETED:
                            with st.chat_message(message.step_role.value):
                                st.markdown(
                                    f"\n:blue[Workflow Step:] {message.step_name}....Completed"
                                )
                                st.session_state.messages.append(
                                    {
                                        "role": message.step_role.value,
                                        "content": f"\n:blue[Workflow Step:] {message.step_name}....Completed",
                                    }
                                )
                        elif message.step_type == MessageType.STEP_DATA:
                            if isinstance(message.content, dict):
                                for key, value in message.content.items():
                                    if key == "risk_report":
                                        data = []
                                        for (
                                            risk_report_key,
                                            risk_report_value,
                                        ) in value.items():
                                            # console.print(
                                            #     f"[bold yellow]Check for {risk_report_key.title()}[/bold yellow]: {pprint(risk_report_key, risk_report_value)}",
                                            #     **message.step_kwargs,
                                            # )
                                            data.append(
                                                f":yellow[Check for {risk_report_key.title()}]: {pprint(risk_report_key, risk_report_value)}"
                                            )

                                        with st.chat_message(message.step_role.value):
                                            st.markdown(data)
                                            st.session_state.messages.append(
                                                {
                                                    "role": message.step_role.value,
                                                    "content": data,
                                                }
                                            )
                                    else:
                                        with st.chat_message(message.step_role.value):
                                            st.markdown(
                                                f":yellow[{key.replace('_', ' ').title()}]: {pprint(key, value)}"
                                            )
                                            st.session_state.messages.append(
                                                {
                                                    "role": message.step_role.value,
                                                    "content": f":yellow[{key.replace('_', ' ').title()}]: {pprint(key, value)}",
                                                }
                                            )
                            else:
                                with st.chat_message(message.step_role.value):
                                    st.markdown(message.content)
                                    st.session_state.messages.append(
                                        {
                                            "role": message.step_role.value,
                                            "content": message.content,
                                        }
                                    )
                    elif event.type == "run.awaiting":
                        if hasattr(event, "run"):
                            message = WorkflowStepMessage(
                                **json.loads(
                                    event.run.await_request.message.parts[0].content
                                )
                            )
                            with st.chat_message(message.step_role.value):
                                st.markdown(f":blue[{message.content}]")
                                st.session_state.messages.append(
                                    {
                                        "role": message.step_role.value,
                                        "content": f":blue[{message.content}]",
                                    }
                                )
                            st.session_state["input_message_type"] = (
                                MessageType.HITL_RESPONSE
                            )
                            st.session_state["input_message_key"] = "response"
                            st.session_state["input_message_query"] = (
                                "Enter your response here"
                            )
                            st.rerun()
                            # COMPLETED = True

                    elif event.type == "run.completed":
                        COMPLETED = True

                if COMPLETED:
                    break


@app.command()
def main(
    host: Annotated[
        str,
        typer.Option(
            help="Please enter GAF Guard Host.",
            rich_help_panel="Hostname",
        ),
    ] = "localhost",
    port: Annotated[
        int,
        typer.Option(
            help="Please enter GAF Guard Port.",
            rich_help_panel="Port",
        ),
    ] = 8000,
):
    os.system("clear")
    asyncio.run(run_app(host=host, port=port))


if __name__ == "__main__":
    app(standalone_mode=False)
