from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import desc
import os
from datetime import datetime, timedelta
from typing import Optional

from database import get_db, PredictionHistory, engine, Base, SessionLocal
from schemas import (
    PredictionRequest, PredictionResponse, HistoryResponse, 
    StationInfo, HealthResponse
)
from prediction_service import init_prediction_service

# Initialize FastAPI
app = FastAPI(
    title="Drought Risk Prediction API",
    description="API untuk prediksi risiko kekeringan berdasarkan data iklim",
    version="1.0.0"
)

# Setup CORS untuk frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Production: ganti dengan domain frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize prediction service
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
prediction_service = None

@app.on_event("startup")
async def startup_event():
    """Initialize saat aplikasi startup"""
    global prediction_service
    try:
        prediction_service = init_prediction_service(PROJECT_DIR)
        Base.metadata.create_all(bind=engine)
        print("✓ Startup selesai - Model dan Database siap")
    except Exception as e:
        print(f"✗ Startup error: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup saat aplikasi shutdown"""
    print("✓ Shutdown selesai")

# ============= ENDPOINTS =============

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check kesehatan API"""
    try:
        db = SessionLocal()
        _ = db.query(PredictionHistory).limit(1).all()
        db.close()
        db_status = True
    except:
        db_status = False
    
    return HealthResponse(
        status="healthy" if prediction_service and db_status else "degraded",
        model_loaded=prediction_service is not None and prediction_service.is_ready(),
        database_connected=db_status,
        message="Semua sistem ready" if prediction_service and db_status else "Ada komponen yang belum siap"
    )

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest, db: Session = Depends(get_db)):
    """
    Prediksi risiko kekeringan untuk satu stasiun
    
    **Input features:**
    - station_id: ID unik stasiun (e.g., "STN001")
    - tn: Temperatur minimum (°C)
    - tx: Temperatur maksimum (°C)
    - tavg: Temperatur rata-rata (°C)
    - rh_avg: Kelembaban rata-rata (%)
    - precipitation: Curah hujan (mm)
    - ss: Durasi sinar matahari (jam)
    - ff_x: Kecepatan angin maksimum (m/s)
    - ddd_car: Arah angin dominan (N, S, E, W, NE, SE, SW, NW)
    """
    
    if not prediction_service or not prediction_service.is_ready():
        raise HTTPException(status_code=503, detail="Model belum ready. Restart server")
    
    try:
        # Prediksi
        prediction = prediction_service.predict(
            tn=request.tn,
            tx=request.tx,
            tavg=request.tavg,
            rh_avg=request.rh_avg,
            precipitation=request.precipitation,
            ss=request.ss,
            ff_x=request.ff_x,
            ddd_car=request.ddd_car
        )
        
        # Simpan ke database
        db_record = PredictionHistory(
            station_id=request.station_id,
            station_name=request.station_name,
            region_name=request.region_name,
            tn=request.tn,
            tx=request.tx,
            tavg=request.tavg,
            rh_avg=request.rh_avg,
            precipitation=request.precipitation,
            ss=request.ss,
            ff_x=request.ff_x,
            ddd_car=request.ddd_car,
            spi=prediction['spi'],
            risk=prediction['risk'],
            risk_score=prediction['risk_score'],
            latitude=request.latitude,
            longitude=request.longitude
        )
        db.add(db_record)
        db.commit()
        db.refresh(db_record)
        
        # Return response
        return PredictionResponse(
            id=db_record.id,
            station_id=request.station_id,
            station_name=request.station_name,
            region_name=request.region_name,
            tn=request.tn,
            tx=request.tx,
            tavg=request.tavg,
            rh_avg=request.rh_avg,
            precipitation=request.precipitation,
            ss=request.ss,
            ff_x=request.ff_x,
            ddd_car=request.ddd_car,
            spi=prediction['spi'],
            risk=prediction['risk'],
            risk_score=prediction['risk_score'],
            latitude=request.latitude,
            longitude=request.longitude,
            created_at=db_record.created_at
        )
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=f"Prediksi gagal: {str(e)}")

@app.post("/predict/quick", tags=["Prediction"])
async def quick_predict(
    rainfall: float,
    temperature: float,
    db: Session = Depends(get_db)
):
    """
    Quick prediction endpoint - hanya perlu curah hujan dan suhu
    (Untuk React frontend)
    
    **Query parameters:**
    - rainfall: Curah hujan (mm)
    - temperature: Suhu rata-rata (°C)
    
    **Response:**
    ```json
    {
      "risk": "Risiko Rendah",
      "spi": 1.234,
      "risk_score": 0.95
    }
    ```
    """
    
    if not prediction_service or not prediction_service.is_ready():
        raise HTTPException(status_code=503, detail="Model belum ready. Restart server")
    
    try:
        # Use default/reasonable values untuk parameters yang tidak ada
        # Asumsi: temperature adalah tavg
        tavg = temperature
        tn = tavg - 2  # Min suhu roughly 2 derajat lebih rendah
        tx = tavg + 3  # Max suhu roughly 3 derajat lebih tinggi
        
        # Default values untuk parameters lain
        rh_avg = 70.0  # Default humidity
        ss = 6.0       # Default sunshine
        ff_x = 5.0     # Default wind speed
        ddd_car = "N"  # Default wind direction
        
        prediction = prediction_service.predict(
            tn=tn,
            tx=tx,
            tavg=tavg,
            rh_avg=rh_avg,
            precipitation=rainfall,
            ss=ss,
            ff_x=ff_x,
            ddd_car=ddd_car
        )
        
        # Convert risk to Indonesian label
        risk_label = prediction['risk']
        if risk_label == 'high':
            risk_indo = 'Risiko Tinggi'
        elif risk_label == 'medium':
            risk_indo = 'Risiko Sedang'
        else:
            risk_indo = 'Risiko Rendah'
        
        # Simpan ke database (optional - buat tracking)
        db_record = PredictionHistory(
            station_id="QUICK",
            station_name="Quick Prediction",
            region_name="Direct",
            tn=tn,
            tx=tx,
            tavg=tavg,
            rh_avg=rh_avg,
            precipitation=rainfall,
            ss=ss,
            ff_x=ff_x,
            ddd_car=ddd_car,
            spi=prediction['spi'],
            risk=risk_label,
            risk_score=prediction['risk_score'],
            latitude=None,
            longitude=None
        )
        db.add(db_record)
        db.commit()
        
        return {
            "risk": risk_indo,
            "spi": prediction['spi'],
            "risk_score": prediction['risk_score'],
            "prediction_type": "quick"
        }
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=f"Prediksi gagal: {str(e)}")

@app.get("/predict/history", response_model=HistoryResponse, tags=["History"])
async def get_history(
    station_id: Optional[str] = None,
    risk: Optional[str] = None,
    days: Optional[int] = 30,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """
    Ambil history prediksi
    
    **Query parameters:**
    - station_id: Filter by stasiun (opsional)
    - risk: Filter by risk level: low, medium, high (opsional)
    - days: Data dari N hari terakhir (default: 30)
    - limit: Max records (default: 100, max: 1000)
    """
    
    limit = min(limit, 1000)  # Max 1000
    
    try:
        query = db.query(PredictionHistory)
        
        # Filter by date
        if days:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            query = query.filter(PredictionHistory.created_at >= cutoff_date)
        
        # Filter by station
        if station_id:
            query = query.filter(PredictionHistory.station_id == station_id)
        
        # Filter by risk level
        if risk and risk in ['low', 'medium', 'high']:
            query = query.filter(PredictionHistory.risk == risk)
        
        # Sort by newest first
        records = query.order_by(desc(PredictionHistory.created_at)).limit(limit).all()
        
        results = [
            PredictionResponse(
                id=r.id,
                station_id=r.station_id,
                station_name=r.station_name,
                region_name=r.region_name,
                tn=r.tn,
                tx=r.tx,
                tavg=r.tavg,
                rh_avg=r.rh_avg,
                precipitation=r.precipitation,
                ss=r.ss,
                ff_x=r.ff_x,
                ddd_car=r.ddd_car,
                spi=r.spi,
                risk=r.risk,
                risk_score=r.risk_score,
                latitude=r.latitude,
                longitude=r.longitude,
                created_at=r.created_at
            )
            for r in records
        ]
        
        return HistoryResponse(total=len(results), data=results)
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Query history gagal: {str(e)}")

@app.get("/predict/history/{station_id}", response_model=HistoryResponse, tags=["History"])
async def get_station_history(
    station_id: str,
    days: int = 30,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """Ambil history prediksi untuk stasiun spesifik"""
    
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        records = (
            db.query(PredictionHistory)
            .filter(
                (PredictionHistory.station_id == station_id) &
                (PredictionHistory.created_at >= cutoff_date)
            )
            .order_by(desc(PredictionHistory.created_at))
            .limit(limit)
            .all()
        )
        
        results = [
            PredictionResponse(
                id=r.id,
                station_id=r.station_id,
                station_name=r.station_name,
                region_name=r.region_name,
                tn=r.tn,
                tx=r.tx,
                tavg=r.tavg,
                rh_avg=r.rh_avg,
                precipitation=r.precipitation,
                ss=r.ss,
                ff_x=r.ff_x,
                ddd_car=r.ddd_car,
                spi=r.spi,
                risk=r.risk,
                risk_score=r.risk_score,
                latitude=r.latitude,
                longitude=r.longitude,
                created_at=r.created_at
            )
            for r in records
        ]
        
        return HistoryResponse(total=len(results), data=results)
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Query history gagal: {str(e)}")

@app.get("/stats/risk-distribution", tags=["Statistics"])
async def risk_distribution(days: int = 30, db: Session = Depends(get_db)):
    """Distribusi risiko dari prediksi terakhir N hari"""
    
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        records = db.query(PredictionHistory).filter(
            PredictionHistory.created_at >= cutoff_date
        ).all()
        
        if not records:
            return {"high": 0, "medium": 0, "low": 0, "total": 0}
        
        risk_counts = {"high": 0, "medium": 0, "low": 0}
        for r in records:
            risk_counts[r.risk] += 1
        
        return {**risk_counts, "total": len(records)}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Stats gagal: {str(e)}")

@app.get("/stats/average-spi", tags=["Statistics"])
async def average_spi(
    days: int = 30,
    region: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Rata-rata SPI dari prediksi terakhir N hari"""
    
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        query = db.query(PredictionHistory).filter(
            PredictionHistory.created_at >= cutoff_date
        )
        
        if region:
            query = query.filter(PredictionHistory.region_name == region)
        
        records = query.all()
        
        if not records:
            return {"average_spi": 0, "count": 0}
        
        avg_spi = sum(r.spi for r in records) / len(records)
        
        return {"average_spi": round(avg_spi, 4), "count": len(records)}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Stats gagal: {str(e)}")

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint - info API"""
    return {
        "title": "Drought Risk Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "POST /predict",
            "history": "GET /predict/history",
            "station_history": "GET /predict/history/{station_id}",
            "risk_stats": "GET /stats/risk-distribution",
            "spi_stats": "GET /stats/average-spi",
            "docs": "/docs (Swagger UI)",
            "redoc": "/redoc (ReDoc)"
        }
    }

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    
    print("=" * 60)
    print("Drought Risk Prediction API")
    print("=" * 60)
    print(f"Project Dir: {PROJECT_DIR}")
    print("Starting server...")
    print("API Documentation: http://localhost:8000/docs")
    print("=" * 60)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
