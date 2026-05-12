from datetime import datetime

from pydantic import BaseModel


class Transaction(BaseModel):
    amount: float
    installments: int
    requested_at: datetime


class Customer(BaseModel):
    avg_amount: float
    tx_count_24h: int
    known_merchants: list[str]


class Merchant(BaseModel):
    id: str
    mcc: str
    avg_amount: float


class Terminal(BaseModel):
    is_online: bool
    card_present: bool
    km_from_home: float


class LastTransaction(BaseModel):
    timestamp: datetime
    km_from_current: float


class FraudScoreRequest(BaseModel):
    id: str
    transaction: Transaction
    customer: Customer
    merchant: Merchant
    terminal: Terminal
    last_transaction: LastTransaction | None


class FraudScoreResponse(BaseModel):
    approved: bool
    fraud_score: float
