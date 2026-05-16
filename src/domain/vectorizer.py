from src.domain.fraud_score import FraudScoreRequest

_MAX_AMOUNT: float = 10_000.0
_MAX_INSTALLMENTS: float = 12.0
_AMOUNT_VS_AVG_RATIO: float = 10.0
_MAX_MINUTES: float = 1_440.0
_MAX_KM: float = 1_000.0
_MAX_TX_COUNT_24H: float = 20.0
_MAX_MERCHANT_AVG_AMOUNT: float = 10_000.0

_MCC_RISK: dict[str, float] = {
    "5411": 0.15,
    "5812": 0.30,
    "5912": 0.20,
    "5944": 0.45,
    "7801": 0.80,
    "7802": 0.75,
    "7995": 0.85,
    "4511": 0.35,
    "5311": 0.25,
    "5999": 0.50,
}


class FraudVectorizer:
    def vectorize(self, request: FraudScoreRequest) -> list[float]:
        tx = request.transaction
        customer = request.customer
        merchant = request.merchant
        terminal = request.terminal
        last = request.last_transaction

        if last is not None:
            delta_minutes = (
                tx.requested_at - last.timestamp
            ).total_seconds() / 60.0
            minutes_since_last_tx = self._clamp(delta_minutes / _MAX_MINUTES)
            km_from_last_tx = self._clamp(last.km_from_current / _MAX_KM)
        else:
            minutes_since_last_tx = -1.0
            km_from_last_tx = -1.0

        return [
            self._clamp(tx.amount / _MAX_AMOUNT),
            self._clamp(tx.installments / _MAX_INSTALLMENTS),
            self._clamp((tx.amount / customer.avg_amount) / _AMOUNT_VS_AVG_RATIO),
            tx.requested_at.hour / 23.0,
            tx.requested_at.weekday() / 6.0,
            minutes_since_last_tx,
            km_from_last_tx,
            self._clamp(terminal.km_from_home / _MAX_KM),
            self._clamp(customer.tx_count_24h / _MAX_TX_COUNT_24H),
            1.0 if terminal.is_online else 0.0,
            1.0 if terminal.card_present else 0.0,
            1.0 if merchant.id not in customer.known_merchants else 0.0,
            _MCC_RISK.get(merchant.mcc, 0.5),
            self._clamp(merchant.avg_amount / _MAX_MERCHANT_AVG_AMOUNT),
        ]

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(1.0, value))
