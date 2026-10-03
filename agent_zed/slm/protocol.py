"""AgentTokenLanguage: Ultra-Efficient Agent-to-Agent Binary/Token Protocol."""

import json
import base64
from typing import Dict, Any, List

class AgentTokenLanguage:
    """Compact tokenized binary protocol for ultra-fast internal CEO-to-Agent communication."""

    # Opcode mapping for agent directives
    OPCODES = {
        "JOB_DISPATCH": 0x01,
        "AST_VERIFY": 0x02,
        "DEBUG_PATCH": 0x03,
        "SEARCH_QUERY": 0x04,
        "OPTIMIZE_LOOP": 0x05,
        "STATUS_OK": 0x06,
        "STATUS_FAIL": 0x07
    }

    REVERSE_OPCODES = {v: k for k, v in OPCODES.items()}

    @classmethod
    def encode_message(cls, agent_id: str, opcode_name: str, payload: Dict[str, Any]) -> str:
        """Encode message into compact agent token language format: [AGENT_ID]::[OPCODE_HEX]::[B64_PAYLOAD]"""
        opcode = cls.OPCODES.get(opcode_name, 0x01)
        json_str = json.dumps(payload)
        b64_payload = base64.b64encode(json_str.encode('utf-8')).decode('utf-8')
        return f"{agent_id}::{opcode:02X}::{b64_payload}"

    @classmethod
    def decode_message(cls, encoded_str: str) -> Dict[str, Any]:
        """Decode compact agent token language back into readable payload."""
        parts = encoded_str.split("::", 2)
        if len(parts) < 3:
            return {"agent_id": "UNKNOWN", "opcode": "JOB_DISPATCH", "payload": {}}

        agent_id, opcode_hex, b64_payload = parts
        opcode_num = int(opcode_hex, 16)
        opcode_name = cls.REVERSE_OPCODES.get(opcode_num, "JOB_DISPATCH")
        json_bytes = base64.b64decode(b64_payload.encode('utf-8'))
        payload = json.loads(json_bytes.decode('utf-8'))

        return {
            "agent_id": agent_id,
            "opcode": opcode_name,
            "payload": payload
        }
