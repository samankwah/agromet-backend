"""Static published copy (FAQ, Terms, Privacy) and the one write path that
needs none of it: a farmer's message from the Contact screen."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from ..database import get_connection
from ..schemas import ContactMessageRequest, ContactMessageResponse, FAQResponse, LegalDocumentResponse, LegalSection

router = APIRouter(tags=["content"])

FAQ_MESSAGES = {
    "when-to-plant-maize": "In Ghana, maize is typically planted at the start of the rains. Target April to June for the major season and September to November for the minor season, depending on local rainfall onset.",
    "when-to-plant-rice": "Rice planting depends on irrigation and region. Rainfed systems usually start with the first dependable rains, while irrigated rice can be staggered year-round.",
    "maize-fertilizer": "Use a soil test where possible. A practical starting point is a balanced basal NPK application followed by a nitrogen top-dress at early vegetative growth.",
    "rainy-season-farming": "Prepare fields early, use drainage where needed, and match planting windows to local rainfall onset instead of fixed calendar dates.",
}

# The legal documents, served as structured sections. Kept here beside
# FAQ_MESSAGES because both are static published copy rather than data.
#
# One source of truth on purpose: this wording previously existed only in the
# web app's TermsOfService.jsx / PrivacyPolicy.jsx, so the mobile app had no way
# to show it without a second copy that would drift. Both clients now render
# these same sections in their own components.
LEGAL_DOCUMENTS = {
    "terms": {
        "title": "Terms of Service",
        "summary": "Please read these terms carefully before using AgroMet.",
        "updated": "April 2026",
        "sections": [
            {
                "title": "Acceptance of Terms",
                "body": "By accessing or using AgroMet, you agree to be bound by these Terms of Service. If you do not agree with any part of these terms, please do not use our services.",
            },
            {
                "title": "User Responsibilities",
                "body": "As a user of AgroMet, you agree to:",
                "items": [
                    "Provide accurate and complete information when creating an account",
                    "Keep your account credentials secure and confidential",
                    "Notify us immediately of any unauthorized access to your account",
                    "Use our services in compliance with all applicable laws and regulations",
                ],
            },
            {
                "title": "Limitation of Liability",
                "body": "AgroMet provides advisories as guidance based on the best available data. Our liability is limited to the fullest extent permitted by law. We are not responsible for any indirect, incidental, or consequential damages resulting from reliance on the service.",
            },
            {
                "title": "Changes to These Terms",
                "body": "We reserve the right to update or modify these Terms at any time. Material changes will be communicated through the platform. Your continued use of AgroMet after changes take effect constitutes acceptance of the updated Terms.",
            },
        ],
    },
    "privacy": {
        "title": "Privacy Policy",
        "summary": "How AgroMet collects, uses and protects your information.",
        "updated": "April 2026",
        "sections": [
            {
                "title": "Information We Collect",
                "body": "We may collect the following types of information:",
                "items": [
                    "Personal identification information (name, email, phone)",
                    "Usage data describing how you interact with our services",
                    "Cookies and similar tracking technologies",
                    "Location data when you opt in to localized advisories",
                ],
            },
            {
                "title": "How We Use Your Information",
                "body": "We use the information we collect to:",
                "items": [
                    "Provide, operate, and maintain the AgroMet platform",
                    "Personalize advisories and recommendations to your location",
                    "Communicate with you about updates, alerts, and support",
                    "Analyze usage patterns to improve the product",
                ],
            },
            {
                "title": "Data Security",
                "body": "We take the security of your personal information seriously and implement administrative, technical, and physical safeguards designed to protect it against unauthorized access, alteration, disclosure, or destruction.",
            },
            {
                "title": "Third-Party Services",
                "body": "We may engage vetted third-party service providers to help us operate and improve AgroMet. These providers have access to your information only to perform tasks on our behalf and are contractually obligated to protect it.",
            },
            # DRAFT, awaiting sign-off. Written because the policy did not say
            # this at all while the app was already doing it: a farmer's typed
            # question, their voice recording and their crop photo all leave
            # Ghana to reach a provider abroad, and "vetted third-party service
            # providers" above does not disclose that in a way anyone could act
            # on. Replace the wording with whatever the agency approves, but do
            # not ship the assistant with nothing here.
            {
                "title": "AgroMet AI and Your Questions",
                "body": "When you ask AgroMet AI a question, record one by voice, or send a crop photo, that content is sent to an artificial intelligence provider outside Ghana to produce the answer. Alongside your question we send the region you have selected in the app and the crops you have listed, so the answer can be specific to your area. We do not send your name, your phone number or your email address.",
                "items": [
                    "Your questions are used to produce your answer, not to identify you",
                    "Conversations are not stored on our servers; they stay on your phone and clear themselves",
                    "Do not include personal details, identity numbers or payment information in a question",
                    "Answers are guidance and can be wrong; check anything critical with your district extension officer",
                ],
            },
            {
                "title": "Changes to This Privacy Policy",
                "body": "We may update this Privacy Policy from time to time. Material changes will be posted on this page with a new effective date. We encourage you to review this policy periodically.",
            },
        ],
    },
}


@router.get("/api/faq/{topic}", response_model=FAQResponse)
def faq(topic: str):
    message = FAQ_MESSAGES.get(topic)
    if not message:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="FAQ topic not found.")
    return FAQResponse(success=True, message=message)


@router.post("/api/contact", response_model=ContactMessageResponse, status_code=status.HTTP_201_CREATED)
def submit_contact_message(payload: ContactMessageRequest):
    """Takes a message from the apps' Contact screen and stores it.

    Deliberately unauthenticated: someone who cannot sign in is exactly the
    person most likely to need to get in touch. Validation lives in the schema.

    The reference returned is the row id, so a follow-up call ("I wrote in on
    Tuesday") can be matched to a record rather than searched for by memory.
    """
    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO contact_messages (name, email, phone, subject, message, source)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                payload.name.strip(),
                payload.email,
                (payload.phone or "").strip() or None,
                payload.subject.strip(),
                payload.message.strip(),
                payload.source.strip() or "mobile",
            ),
        )
        reference = cursor.lastrowid

    return ContactMessageResponse(
        success=True,
        message="Thank you. Your message has reached the AgroMet team.",
        reference=reference,
    )


@router.get("/api/legal/{slug}", response_model=LegalDocumentResponse)
def legal_document(slug: str):
    document = LEGAL_DOCUMENTS.get(slug)
    if not document:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Legal document not found.")

    return LegalDocumentResponse(
        success=True,
        slug=slug,
        title=document["title"],
        summary=document["summary"],
        updated=document["updated"],
        sections=[LegalSection(**section) for section in document["sections"]],
    )
