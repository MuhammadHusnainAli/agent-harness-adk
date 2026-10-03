"""OpenAPI → tools: hand an agent an API by handing it the API's document.

A small order API runs here on a real socket, and serves its own
`openapi.json`. The agent is given nothing but that URL: every operation in the
document becomes a tool, with the arguments the document describes.
"""

from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from _common import pick_provider

from agent_harness import Agent, Harness, openapi_tools, tool_call

ORDERS = {"4182": {"id": "4182", "status": "shipped", "total": 40, "customer": "c_17"},
          "4190": {"id": "4190", "status": "open", "total": 15, "customer": "c_17"}}

SPEC = {
    "openapi": "3.0.3",
    "info": {"title": "Orders API", "version": "1.0.0"},
    "servers": [{"url": "/v1"}],
    "paths": {
        "/orders": {"get": {
            "operationId": "listOrders", "summary": "List a customer's orders",
            "parameters": [
                {"name": "customer", "in": "query", "required": True,
                 "description": "The customer id.", "schema": {"type": "string"}},
                {"name": "status", "in": "query",
                 "schema": {"type": "string", "enum": ["open", "shipped", "refunded"]}}],
            "responses": {"200": {"description": "The orders."}}}},
        "/orders/{orderId}": {"get": {
            "operationId": "getOrder", "summary": "Get one order",
            "parameters": [{"name": "orderId", "in": "path", "required": True,
                            "schema": {"type": "string"}}],
            "responses": {"200": {"description": "The order."}}}},
        "/orders/{orderId}/refunds": {"post": {
            "operationId": "refundOrder", "summary": "Refund an order, in part or whole",
            "parameters": [{"name": "orderId", "in": "path", "required": True,
                            "schema": {"type": "string"}}],
            "requestBody": {"required": True, "content": {"application/json": {
                "schema": {"$ref": "#/components/schemas/Refund"}}}},
            "responses": {"201": {"description": "The refund."}}}},
    },
    "components": {
        "schemas": {"Refund": {"type": "object", "required": ["amount", "reason"],
                               "properties": {
            "amount": {"type": "number", "description": "How much, in EUR."},
            "reason": {"type": "string", "enum": ["duplicate", "damaged", "other"]}}}},
        "securitySchemes": {"key": {"type": "apiKey", "in": "header", "name": "X-API-Key"}},
    },
}


class Api(BaseHTTPRequestHandler):
    def _reply(self, status: int, body: Any) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        path, _, query = self.path.partition("?")
        if path == "/openapi.json":
            return self._reply(200, SPEC)
        if self.headers.get("X-API-Key") != "demo-key":
            return self._reply(401, {"error": "an API key is required"})
        if path == "/v1/orders":
            wanted = dict(pair.split("=") for pair in query.split("&") if pair)
            return self._reply(200, [o for o in ORDERS.values()
                                     if o["customer"] == wanted.get("customer")
                                     and wanted.get("status", o["status"]) == o["status"]])
        order = ORDERS.get(path.rsplit("/", 1)[-1])
        self._reply(200, order) if order else self._reply(404, {"error": "no such order"})

    def do_POST(self) -> None:  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        order = self.path.split("/")[3]
        ORDERS[order]["status"] = "refunded"
        self._reply(201, {"order": order, "refunded": body["amount"],
                          "reason": body["reason"]})

    def log_message(self, *args: Any) -> None:
        return None


async def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), Api)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}"

    # --- the document becomes tools ------------------------------------------
    api = openapi_tools(f"{url}/openapi.json", api_key="demo-key", writes="ask")
    print(api.describe(), "\n")

    # --- call one yourself, to see the API is wired up -------------------------
    print("direct     ", await api.call("getOrder", orderId="4190"), "\n")

    # --- and hand them to an agent ----------------------------------------------
    provider, model = pick_provider([
        tool_call("listOrders", customer="c_17", status="shipped"),
        tool_call("refundOrder", orderId="4182", amount=40, reason="duplicate"),
        "Order 4182 was charged twice, so I refunded the 40 EUR.",
    ])
    harness = Harness(provider=provider)
    # `writes="ask"` means a refund needs approving. Here, everything is.
    harness.policy.approver = lambda tool, args, reason: (
        print(f"approve    {tool}({args})") or True)
    agent = Agent("support", "Help the customer with their orders. Look an order "
                  "up before you change it.", tools=api, model=model,
                  harness=harness, memory=False)
    async for event in agent.stream("Customer c_17 was charged twice for a shipped "
                                    "order. Refund the duplicate."):
        if event.type == "tool_result":
            print(f"tool       {event.data['tool']} → "
                  f"{' '.join(event.text.split())[:90]}")
        elif event.type == "run_end":
            print(f"\nsupport    {event.data['result'].output}")

    await api.aclose()
    await harness.aclose()
    server.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
