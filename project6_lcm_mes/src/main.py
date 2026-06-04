from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes import auth, production, equipment, quality, material, warehouse, reports, ai
from src.websocket.manager import manager
from src.core.config import settings

app = FastAPI(
    title="LCM MES System API",
    description="Liquid Crystal Module Manufacturing Execution System",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(production.router)
app.include_router(equipment.router)
app.include_router(quality.router)
app.include_router(material.router)
app.include_router(warehouse.router)
app.include_router(reports.router)
app.include_router(ai.router)


@app.get("/")
def root():
    return {"message": "LCM MES System API", "version": "1.0.0"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.websocket("/ws/production")
async def websocket_production(websocket: WebSocket):
    await manager.connect(websocket, "production")
    try:
        while True:
            data = await websocket.receive_text()
            await manager.broadcast({"message": data}, "production")
    except WebSocketDisconnect:
        manager.disconnect(websocket, "production")


@app.websocket("/ws/equipment")
async def websocket_equipment(websocket: WebSocket):
    await manager.connect(websocket, "equipment")
    try:
        while True:
            data = await websocket.receive_text()
            await manager.broadcast({"message": data}, "equipment")
    except WebSocketDisconnect:
        manager.disconnect(websocket, "equipment")


@app.websocket("/ws/quality")
async def websocket_quality(websocket: WebSocket):
    await manager.connect(websocket, "quality")
    try:
        while True:
            data = await websocket.receive_text()
            await manager.broadcast({"message": data}, "quality")
    except WebSocketDisconnect:
        manager.disconnect(websocket, "quality")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
