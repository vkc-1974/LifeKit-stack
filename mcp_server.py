#!/bin/env python3

# mcp_server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mariadb
import uvicorn

app = FastAPI(title="MCP Server")

# DB connection
conn = mariadb.connect(host='mariadb',
                       port=3306,
                       user='livekit_user',
                       password='livekit_pass',
                       database='livekit_db')
cur = conn.cursor()

class BalanceRequest(BaseModel):
    user_id: int

@app.post("/tools/get_user_balance")
async def get_user_balance(req: BalanceRequest):
    try:
        cur.execute("SELECT balance_value FROM users WHERE user_id = ?", (req.user_id,))
        row = cur.fetchone()
        if row:
            return {"content": f"Баланс пользователя {req.user_id}: {row[0]}"}
        else:
            return {"content": f"Пользователь {req.user_id} не найден"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
