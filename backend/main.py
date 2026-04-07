"""
PayRoute AI - Backend API
Intelligent Payment Gateway Routing using Gemini + local knowledge context
"""

import os
import json
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import google.generativeai as genai

load_dotenv()

app = FastAPI(title="PayRoute AI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Global State ───────────────────────────────────────────────────────────
knowledge_context = ""
qa_chain = None

KNOWLEDGE_BASE_DIR = Path(__file__).parent.parent / "knowledge_base"
FRONTEND_INDEX = Path(__file__).parent.parent / "public" / "index.html"


def _clear_broken_proxy_env() -> None:
    # Some Windows environments inject dead localhost proxy settings that break
    # Gemini API calls during startup. Clear them for this process only.
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        os.environ.pop(key, None)


# ─── Pydantic Models ─────────────────────────────────────────────────────────

class TransactionRequest(BaseModel):
    amount: float
    currency: str = "INR"
    country: str = "India"
    merchant_category: str
    payment_method_preference: Optional[str] = "any"
    transaction_type: str = "one_time"   # one_time | subscription | marketplace | payout
    priority: str = "balanced"           # cost | speed | success_rate | balanced
    notes: Optional[str] = ""


class GatewayRecommendation(BaseModel):
    gateway: str
    score: float
    rank: int
    estimated_fee: str
    success_rate: str
    settlement_time: str
    key_reasons: list[str]
    warnings: list[str]


class RouteResponse(BaseModel):
    recommendations: list[GatewayRecommendation]
    summary: str
    transaction_context: dict
    rag_context_used: str


# ─── Startup: Build Vector Store ─────────────────────────────────────────────

@app.on_event("startup")
async def load_knowledge_base():
    global knowledge_context, qa_chain
    _clear_broken_proxy_env()

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("WARNING: GOOGLE_API_KEY not set - running in demo mode")
        return

    print("Loading knowledge base...")
    sections = []
    for md_file in KNOWLEDGE_BASE_DIR.glob("*.md"):
        sections.append(f"# Source: {md_file.name}\n{md_file.read_text(encoding='utf-8')}")
        print(f"  Loaded {md_file.name}")

    knowledge_context = "\n\n---\n\n".join(sections)
    genai.configure(api_key=api_key)
    qa_chain = genai.GenerativeModel(os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
    print("Gemini model ready")


def _build_prompt(query: str) -> str:
    return f"""
You are PayRoute AI, an expert payment gateway advisor for Indian and global payment infrastructure.

Using the retrieved knowledge base context below, recommend the best payment gateways for this transaction.

CONTEXT FROM KNOWLEDGE BASE:
{knowledge_context}

TRANSACTION DETAILS:
{query}

Respond ONLY with a valid JSON object (no markdown, no explanation outside JSON) in this exact structure:
{{
  "recommendations": [
    {{
      "gateway": "Gateway Name",
      "score": 9.2,
      "rank": 1,
      "estimated_fee": "2% = ₹XXX",
      "success_rate": "94-97%",
      "settlement_time": "T+2 days",
      "key_reasons": ["Reason 1", "Reason 2", "Reason 3"],
      "warnings": ["Any concern if applicable"]
    }}
  ],
  "summary": "2-3 sentence plain English explanation of the recommendation",
  "rag_context_used": "Which specific facts from the knowledge base influenced this decision"
}}

Include exactly 3 ranked gateways. Be specific about fees using the actual transaction amount provided.
"""


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def frontend():
    if FRONTEND_INDEX.exists():
        return FRONTEND_INDEX.read_text(encoding="utf-8")
    raise HTTPException(status_code=404, detail="Frontend not found")


@app.get("/api/health")
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "vectorstore_ready": bool(knowledge_context),
        "chain_ready": qa_chain is not None,
    }


@app.post("/api/route", response_model=RouteResponse)
@app.post("/route", response_model=RouteResponse)
async def route_payment(req: TransactionRequest):
    if qa_chain is None:
        # Demo mode: return mock data
        return _demo_response(req)

    query = f"""
Transaction Details:
- Amount: {req.currency} {req.amount:,.2f}
- Country: {req.country}
- Merchant Category: {req.merchant_category}
- Payment Method Preference: {req.payment_method_preference}
- Transaction Type: {req.transaction_type}
- Optimization Priority: {req.priority}
- Additional Notes: {req.notes or 'None'}

Please analyze all relevant gateways and recommend the top 3 in ranked order.
Consider fees, success rates, settlement timelines, compliance, and the merchant category.
"""

    try:
        result = qa_chain.generate_content(
            _build_prompt(query),
            generation_config={"temperature": 0.2},
        )
        raw = result.text.strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        parsed = json.loads(raw)

        return RouteResponse(
            recommendations=[GatewayRecommendation(**r) for r in parsed["recommendations"]],
            summary=parsed.get("summary", ""),
            transaction_context={
                "amount": req.amount,
                "currency": req.currency,
                "country": req.country,
                "category": req.merchant_category,
                "priority": req.priority,
                "type": req.transaction_type,
            },
            rag_context_used=parsed.get("rag_context_used", ""),
        )
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse LLM response: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/gateways")
@app.get("/gateways")
async def list_gateways():
    """Return list of gateways in the knowledge base"""
    gateways = [f.stem for f in KNOWLEDGE_BASE_DIR.glob("*.md")]
    return {"gateways": gateways}


# ─── Demo Mode (no API key needed) ──────────────────────────────────────────

def _demo_response(req: TransactionRequest) -> RouteResponse:
    """
    Returns realistic mock recommendations based on simple rules.
    Used when GOOGLE_API_KEY is not set.
    """
    amount = req.amount
    is_india = "india" in req.country.lower()
    is_upi = "upi" in (req.payment_method_preference or "").lower()
    is_sub = req.transaction_type == "subscription"
    is_payout = req.transaction_type == "payout"
    is_intl = not is_india
    priority = req.priority

    if is_intl:
        recs = [
            GatewayRecommendation(
                gateway="Stripe", score=9.4, rank=1,
                estimated_fee=f"3.9% + $0.30 = ${amount * 0.039 + 0.30:.2f}",
                success_rate="88-94%", settlement_time="T+2 business days",
                key_reasons=[
                    "Best global payment method coverage for international transactions",
                    "Supports 135+ currencies with automatic conversion",
                    "Stripe Radar ML fraud detection reduces chargebacks"
                ],
                warnings=["Currency conversion adds 1.5% for non-USD transactions"]
            ),
            GatewayRecommendation(
                gateway="Razorpay", score=7.1, rank=2,
                estimated_fee=f"3% = ₹{amount * 0.03:,.2f}",
                success_rate="82-88%", settlement_time="T+3 to T+7",
                key_reasons=[
                    "Can process international cards via Indian entity",
                    "Good fallback for India-origin international transactions"
                ],
                warnings=["Not ideal as primary gateway for fully international business"]
            ),
            GatewayRecommendation(
                gateway="PayU", score=6.8, rank=3,
                estimated_fee=f"3-4.5% = ₹{amount * 0.04:,.2f}",
                success_rate="80-87%", settlement_time="T+5 to T+10",
                key_reasons=[
                    "Multi-regional support (LatAm, Eastern Europe, Africa)",
                    "Good for businesses expanding beyond India"
                ],
                warnings=["Limited developer ecosystem compared to Stripe"]
            ),
        ]
        summary = "For international transactions, Stripe is the clear winner with its global payment method coverage and high success rates. Consider Razorpay as a secondary gateway for India-connected international flows."
    elif is_payout:
        recs = [
            GatewayRecommendation(
                gateway="Cashfree", score=9.6, rank=1,
                estimated_fee="₹2-5 per payout (bulk pricing)",
                success_rate="99%+ payout success", settlement_time="Real-time to T+1",
                key_reasons=[
                    "Best-in-class bulk payout infrastructure in India",
                    "Supports 150+ country payouts for international disbursement",
                    "UPI, IMPS, NEFT, RTGS all supported"
                ],
                warnings=["Verify beneficiary KYC for high-value payouts"]
            ),
            GatewayRecommendation(
                gateway="Razorpay", score=8.2, rank=2,
                estimated_fee="₹3-7 per payout",
                success_rate="98% payout success", settlement_time="T+1",
                key_reasons=[
                    "Razorpay Route excellent for marketplace payouts",
                    "Strong API for automated disbursements"
                ],
                warnings=["Higher per-payout fee for small volumes"]
            ),
            GatewayRecommendation(
                gateway="PayU", score=7.0, rank=3,
                estimated_fee="₹5-10 per payout",
                success_rate="97% payout success", settlement_time="T+2",
                key_reasons=[
                    "Reliable for enterprise payout volumes",
                    "Good for insurance claim payouts"
                ],
                warnings=["Slower settlement and higher fees vs Cashfree"]
            ),
        ]
        summary = "For payout/disbursement use cases, Cashfree has the strongest infrastructure in India with real-time settlements and bulk payout APIs."
    elif is_upi or amount < 5000:
        recs = [
            GatewayRecommendation(
                gateway="Cashfree", score=9.5, rank=1,
                estimated_fee="0% (UPI zero MDR)",
                success_rate="95-98%", settlement_time="T+1 to same-day",
                key_reasons=[
                    "Highest UPI success rate in the industry (95-98%)",
                    "Zero MDR on UPI — no transaction cost",
                    "Fastest settlement cycle in India"
                ],
                warnings=[]
            ),
            GatewayRecommendation(
                gateway="Razorpay", score=9.1, rank=2,
                estimated_fee="0% (UPI zero MDR)",
                success_rate="94-97%", settlement_time="T+1",
                key_reasons=[
                    "Excellent UPI success rates with smart retry logic",
                    "Best developer experience for UPI integration",
                    "Razorpay Magic Checkout improves conversion"
                ],
                warnings=[]
            ),
            GatewayRecommendation(
                gateway="PayU", score=7.8, rank=3,
                estimated_fee="0% (UPI zero MDR)",
                success_rate="92-96%", settlement_time="T+2",
                key_reasons=[
                    "Reliable UPI processing for established businesses",
                    "Good for high-volume utility payments via UPI"
                ],
                warnings=["Slightly lower UPI success rate vs Cashfree/Razorpay"]
            ),
        ]
        summary = "For UPI-heavy or low-value transactions in India, Cashfree and Razorpay are optimal. Both have zero MDR on UPI and strong success rates."
    elif priority == "cost":
        recs = [
            GatewayRecommendation(
                gateway="Cashfree", score=9.3, rank=1,
                estimated_fee=f"1.75% = ₹{amount * 0.0175:,.2f}",
                success_rate="89-94%", settlement_time="Instant to T+2",
                key_reasons=[
                    "Lowest MDR rates in India (1.75% vs industry 2%+)",
                    "Zero MDR on UPI reduces blended cost significantly",
                    "Instant settlement option reduces working capital needs"
                ],
                warnings=[]
            ),
            GatewayRecommendation(
                gateway="PayU", score=8.5, rank=2,
                estimated_fee=f"1.99% = ₹{amount * 0.0199:,.2f}",
                success_rate="87-92%", settlement_time="T+2",
                key_reasons=[
                    "Competitive volume-based pricing for INR 10L+ merchants",
                    "Negotiable rates for high-volume transactions"
                ],
                warnings=["Volume discounts require minimum monthly transactions"]
            ),
            GatewayRecommendation(
                gateway="Razorpay", score=7.9, rank=3,
                estimated_fee=f"2.0% = ₹{amount * 0.02:,.2f}",
                success_rate="88-93%", settlement_time="T+2",
                key_reasons=[
                    "Transparent pricing with no hidden fees",
                    "Good balance of cost and reliability"
                ],
                warnings=["Higher fee vs Cashfree for same transaction types"]
            ),
        ]
        summary = "For cost optimization, Cashfree offers the lowest MDR in India. On high UPI volume, all three gateways are zero MDR — so blended cost depends on your card/NB mix."
    else:
        recs = [
            GatewayRecommendation(
                gateway="Razorpay", score=9.3, rank=1,
                estimated_fee=f"2.0% = ₹{amount * 0.02:,.2f}",
                success_rate="88-93%", settlement_time="T+2 (instant available)",
                key_reasons=[
                    "Best developer experience and API documentation in India",
                    "Razorpay Shield AI fraud detection",
                    "Strong subscription and recurring billing support",
                    f"Ideal for {req.merchant_category} category"
                ],
                warnings=[]
            ),
            GatewayRecommendation(
                gateway="Cashfree", score=8.8, rank=2,
                estimated_fee=f"1.75% = ₹{amount * 0.0175:,.2f}",
                success_rate="89-94%", settlement_time="Instant to T+2",
                key_reasons=[
                    "Lower fees than Razorpay with comparable success rates",
                    "Fastest settlement in India",
                    "Good for businesses sensitive to cash flow timing"
                ],
                warnings=[]
            ),
            GatewayRecommendation(
                gateway="PayU", score=7.6, rank=3,
                estimated_fee=f"1.99% = ₹{amount * 0.0199:,.2f}",
                success_rate="87-92%", settlement_time="T+2",
                key_reasons=[
                    "Strong enterprise support and account management",
                    "Good EMI and BNPL product for higher ticket sizes"
                ],
                warnings=["Less developer-friendly than Razorpay/Cashfree"]
            ),
        ]
        summary = f"For a balanced {req.merchant_category} transaction of ₹{amount:,.2f}, Razorpay leads on reliability and developer experience. Cashfree is a strong alternative with lower fees."

    return RouteResponse(
        recommendations=recs,
        summary=summary,
        transaction_context={
            "amount": req.amount,
            "currency": req.currency,
            "country": req.country,
            "category": req.merchant_category,
            "priority": req.priority,
            "type": req.transaction_type,
        },
        rag_context_used="[Demo mode — connect a Gemini API key for full RAG-powered recommendations]",
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
