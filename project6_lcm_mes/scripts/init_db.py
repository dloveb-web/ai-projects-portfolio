import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database.base import Base
from src.database.session import engine, SessionLocal
from src.database.models import User, Material
from src.core.security import get_password_hash


def init_db():
    Base.metadata.create_all(bind=engine)
    
    db = SessionLocal()
    try:
        admin = db.query(User).filter(User.username == "admin").first()
        if not admin:
            admin = User(
                username="admin",
                password_hash=get_password_hash("admin123"),
                role="admin",
                department="工程部",
                name="系统管理员"
            )
            db.add(admin)
        
        manager = db.query(User).filter(User.username == "manager").first()
        if not manager:
            manager = User(
                username="manager",
                password_hash=get_password_hash("manager123"),
                role="production_manager",
                department="PMC",
                name="生产经理"
            )
            db.add(manager)
        
        operator = db.query(User).filter(User.username == "operator").first()
        if not operator:
            operator = User(
                username="operator",
                password_hash=get_password_hash("operator123"),
                role="operator",
                department="生产部",
                name="操作员"
            )
            db.add(operator)
        
        inspector = db.query(User).filter(User.username == "inspector").first()
        if not inspector:
            inspector = User(
                username="inspector",
                password_hash=get_password_hash("inspector123"),
                role="quality_inspector",
                department="品质部",
                name="质检员"
            )
            db.add(inspector)
        
        warehouse = db.query(User).filter(User.username == "warehouse").first()
        if not warehouse:
            warehouse = User(
                username="warehouse",
                password_hash=get_password_hash("warehouse123"),
                role="warehouse_operator",
                department="货仓",
                name="仓管员"
            )
            db.add(warehouse)

        materials = [
            {"code": "POL-001", "name": "偏光片", "unit": "片", "min_stock": 1000, "max_stock": 10000},
            {"code": "REF-001", "name": "反光片", "unit": "片", "min_stock": 1000, "max_stock": 10000},
            {"code": "IC-001", "name": "驱动IC", "unit": "个", "min_stock": 500, "max_stock": 5000},
            {"code": "FPC-001", "name": "柔性电路板", "unit": "片", "min_stock": 500, "max_stock": 5000},
            {"code": "GLASS-001", "name": "玻璃盖板", "unit": "片", "min_stock": 500, "max_stock": 5000},
            {"code": "BL-001", "name": "背光模组", "unit": "个", "min_stock": 500, "max_stock": 5000},
            {"code": "GLUE-001", "name": "水胶", "unit": "ml", "min_stock": 10000, "max_stock": 100000},
            {"code": "ACF-001", "name": "导电胶", "unit": "卷", "min_stock": 100, "max_stock": 1000},
        ]
        
        for mat in materials:
            existing = db.query(Material).filter(Material.code == mat["code"]).first()
            if not existing:
                material = Material(**mat)
                db.add(material)
        
        db.commit()
        print("Database initialized successfully!")
        print("Users created:")
        print("- admin / admin123 (工程部)")
        print("- manager / manager123 (PMC)")
        print("- operator / operator123 (生产部)")
        print("- inspector / inspector123 (品质部)")
        print("- warehouse / warehouse123 (货仓)")
        print("\nMaterials created:")
        for mat in materials:
            print(f"- {mat['code']}: {mat['name']}")
    finally:
        db.close()


if __name__ == "__main__":
    init_db()
