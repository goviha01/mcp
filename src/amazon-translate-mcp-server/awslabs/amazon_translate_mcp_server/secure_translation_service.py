"""
Security-integrated Translation Service for Amazon Translate MCP Server.

This module demonstrates how to integrate security features into translation services,
including input validation, PII detection, audit logging, and secure error handling.
"""

import logging
import time
from typing import List, Optional, Dict, Any, Tuple

from .models import TranslationResult, LanguageDetectionResult, ValidationResult
from .exceptions import ValidationError, SecurityError, TranslationError
from .security import SecurityManager, SecurityConfig
from .config import get_config
from .aws_client import AWSClientManager


logger = logging.getLogger(__name__)


class SecureTranslationService:
    """
    Security-integrated translation service with comprehensive security features.
    
    This service demonstrates how to integrate all security features including:
    - Input validation and sanitization
    - PII detection and masking
    - Content filtering
    - Audit logging
    - Secure error handling
    """
    
    def __init__(self, aws_client_manager: Optional[AWSClientManager] = None):
        """Initialize the secure translation service."""
        self.config = get_config()
        self.security_manager = SecurityManager(self.config.to_security_config())
        self.aws_client_manager = aws_client_manager or AWSClientManager()
        self.logger = logging.getLogger(__name__)
    
    def translate_text_secure(
        self,
        text: str,
        source_language: str,
        target_language: str,
        terminology_names: Optional[List[str]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> Tuple[TranslationResult, List[Dict[str, Any]]]:
        """
        Translate text with comprehensive security features.
        
        Args:
            text: Text to translate
            source_language: Source language code
            target_language: Target language code
            terminology_names: Optional terminology names to use
            user_id: Optional user identifier for audit logging
            session_id: Optional session identifier for audit logging
            
        Returns:
            Tuple of (TranslationResult, PII findings list)
            
        Raises:
            ValidationError: If input validation fails
            SecurityError: If security violations are detected
            TranslationError: If translation operation fails
        """
        operation_start = time.time()
        pii_findings = []
        
        try:
            # Step 1: Input validation and sanitization
            self.logger.debug(f"Validating translation request for {source_language} -> {target_language}")
            
            # Validate and sanitize text input
            sanitized_text, pii_findings = self.security_manager.validate_and_sanitize_text(text, "text")
            
            # Validate language codes
            validated_source = self.security_manager.validator.validate_language_code(
                source_language, "source_language"
            )
            validated_target = self.security_manager.validator.validate_language_code(
                target_language, "target_language"
            )
            
            # Validate terminology names if provided
            validated_terminology = []
            if terminology_names:
                validated_terminology = self.security_manager.validator.validate_terminology_names(
                    terminology_names, "terminology_names"
                )
            
            # Step 2: Log the operation start
            self.security_manager.log_operation(
                operation="translate_text_start",
                source_language=validated_source,
                target_language=validated_target,
                text_length=len(sanitized_text),
                terminology_used=validated_terminology,
                user_id=user_id,
                success=True
            )
            
            # Step 3: Perform translation (simulated - would use actual AWS Translate)
            # In real implementation, this would call AWS Translate service
            translation_result = self._perform_translation(
                sanitized_text,
                validated_source,
                validated_target,
                validated_terminology
            )
            
            # Step 4: Validate translation result
            if pii_findings:
                # Add metadata about PII handling
                translation_result.applied_terminologies.append("PII_MASKED")
            
            # Step 5: Log successful operation
            operation_time = time.time() - operation_start
            self.security_manager.log_operation(
                operation="translate_text_complete",
                source_language=validated_source,
                target_language=validated_target,
                text_length=len(sanitized_text),
                terminology_used=validated_terminology,
                user_id=user_id,
                success=True
            )
            
            self.logger.info(
                f"Translation completed successfully in {operation_time:.2f}s "
                f"({validated_source} -> {validated_target})"
            )
            
            return translation_result, pii_findings
            
        except (ValidationError, SecurityError) as e:
            # Log security/validation errors
            self.security_manager.log_operation(
                operation="translate_text_error",
                source_language=source_language,
                target_language=target_language,
                text_length=len(text) if text else 0,
                user_id=user_id,
                success=False,
                error_code=e.error_code
            )
            
            self.logger.warning(f"Translation validation failed: {e.message}")
            raise
            
        except Exception as e:
            # Log unexpected errors
            self.security_manager.log_operation(
                operation="translate_text_error",
                source_language=source_language,
                target_language=target_language,
                text_length=len(text) if text else 0,
                user_id=user_id,
                success=False,
                error_code="UNEXPECTED_ERROR"
            )
            
            self.logger.error(f"Unexpected error in translation: {str(e)}")
            raise TranslationError(
                f"Translation failed due to unexpected error: {str(e)}",
                source_language=source_language,
                target_language=target_language
            )
    
    def detect_language_secure(
        self,
        text: str,
        user_id: Optional[str] = None
    ) -> Tuple[LanguageDetectionResult, List[Dict[str, Any]]]:
        """
        Detect language with security features.
        
        Args:
            text: Text to analyze
            user_id: Optional user identifier for audit logging
            
        Returns:
            Tuple of (LanguageDetectionResult, PII findings list)
        """
        try:
            # Validate and sanitize input
            sanitized_text, pii_findings = self.security_manager.validate_and_sanitize_text(text, "text")
            
            # Log operation
            self.security_manager.log_operation(
                operation="detect_language",
                text_length=len(sanitized_text),
                user_id=user_id,
                success=True
            )
            
            # Perform language detection (simulated)
            result = self._perform_language_detection(sanitized_text)
            
            self.logger.info(f"Language detected: {result.detected_language} (confidence: {result.confidence_score})")
            
            return result, pii_findings
            
        except (ValidationError, SecurityError) as e:
            self.security_manager.log_operation(
                operation="detect_language_error",
                text_length=len(text) if text else 0,
                user_id=user_id,
                success=False,
                error_code=e.error_code
            )
            raise
    
    def validate_translation_secure(
        self,
        original_text: str,
        translated_text: str,
        source_language: str,
        target_language: str,
        user_id: Optional[str] = None
    ) -> ValidationResult:
        """
        Validate translation quality with security features.
        
        Args:
            original_text: Original text
            translated_text: Translated text
            source_language: Source language code
            target_language: Target language code
            user_id: Optional user identifier for audit logging
            
        Returns:
            ValidationResult with quality assessment
        """
        try:
            # Validate inputs
            sanitized_original, _ = self.security_manager.validate_and_sanitize_text(
                original_text, "original_text"
            )
            sanitized_translated, _ = self.security_manager.validate_and_sanitize_text(
                translated_text, "translated_text"
            )
            
            validated_source = self.security_manager.validator.validate_language_code(
                source_language, "source_language"
            )
            validated_target = self.security_manager.validator.validate_language_code(
                target_language, "target_language"
            )
            
            # Log operation
            self.security_manager.log_operation(
                operation="validate_translation",
                source_language=validated_source,
                target_language=validated_target,
                text_length=len(sanitized_original),
                user_id=user_id,
                success=True
            )
            
            # Perform validation (simulated)
            result = self._perform_translation_validation(
                sanitized_original,
                sanitized_translated,
                validated_source,
                validated_target
            )
            
            self.logger.info(f"Translation validation completed: quality_score={result.quality_score}")
            
            return result
            
        except (ValidationError, SecurityError) as e:
            self.security_manager.log_operation(
                operation="validate_translation_error",
                source_language=source_language,
                target_language=target_language,
                user_id=user_id,
                success=False,
                error_code=e.error_code
            )
            raise
    
    def _perform_translation(
        self,
        text: str,
        source_language: str,
        target_language: str,
        terminology_names: List[str]
    ) -> TranslationResult:
        """
        Perform the actual translation (simulated implementation).
        
        In a real implementation, this would call AWS Translate service.
        """
        # Simulated translation result
        if source_language == "en" and target_language == "es":
            translated_text = "Texto traducido simulado"
        elif source_language == "es" and target_language == "en":
            translated_text = "Simulated translated text"
        else:
            translated_text = f"[Translated from {source_language} to {target_language}]: {text}"
        
        return TranslationResult(
            translated_text=translated_text,
            source_language=source_language,
            target_language=target_language,
            applied_terminologies=terminology_names,
            confidence_score=0.95
        )
    
    def _perform_language_detection(self, text: str) -> LanguageDetectionResult:
        """
        Perform language detection (simulated implementation).
        
        In a real implementation, this would call AWS Translate service.
        """
        # Simple heuristic for simulation
        if any(word in text.lower() for word in ['hello', 'the', 'and', 'is', 'are']):
            detected_language = "en"
            confidence = 0.9
        elif any(word in text.lower() for word in ['hola', 'el', 'la', 'es', 'son']):
            detected_language = "es"
            confidence = 0.85
        else:
            detected_language = "en"  # Default
            confidence = 0.6
        
        return LanguageDetectionResult(
            detected_language=detected_language,
            confidence_score=confidence,
            alternative_languages=[("fr", 0.1), ("de", 0.05)]
        )
    
    def _perform_translation_validation(
        self,
        original_text: str,
        translated_text: str,
        source_language: str,
        target_language: str
    ) -> ValidationResult:
        """
        Perform translation validation (simulated implementation).
        
        In a real implementation, this would use various quality metrics.
        """
        # Simple validation based on length ratio
        length_ratio = len(translated_text) / len(original_text) if original_text else 0
        
        if 0.5 <= length_ratio <= 2.0:
            quality_score = 0.9
            is_valid = True
            issues = []
            suggestions = []
        else:
            quality_score = 0.4
            is_valid = False
            issues = ["Unusual length ratio between original and translated text"]
            suggestions = ["Review translation for completeness"]
        
        return ValidationResult(
            is_valid=is_valid,
            quality_score=quality_score,
            issues=issues,
            suggestions=suggestions
        )


# Example usage and integration patterns
class SecurityIntegrationExample:
    """
    Example class showing various security integration patterns.
    """
    
    def __init__(self):
        self.secure_service = SecureTranslationService()
    
    def example_secure_translation_workflow(self):
        """
        Example of a complete secure translation workflow.
        """
        try:
            # Example 1: Basic secure translation
            text = "Hello, my email is john.doe@example.com"
            result, pii_findings = self.secure_service.translate_text_secure(
                text=text,
                source_language="en",
                target_language="es",
                user_id="user123"
            )
            
            print(f"Translation: {result.translated_text}")
            print(f"PII findings: {len(pii_findings)}")
            
            # Example 2: Language detection with PII
            detect_result, pii_findings = self.secure_service.detect_language_secure(
                text="Contact me at 555-123-4567",
                user_id="user123"
            )
            
            print(f"Detected language: {detect_result.detected_language}")
            print(f"PII masked: {len(pii_findings) > 0}")
            
            # Example 3: Translation validation
            validation_result = self.secure_service.validate_translation_secure(
                original_text="Hello world",
                translated_text="Hola mundo",
                source_language="en",
                target_language="es",
                user_id="user123"
            )
            
            print(f"Translation valid: {validation_result.is_valid}")
            print(f"Quality score: {validation_result.quality_score}")
            
        except ValidationError as e:
            print(f"Validation error: {e.message}")
        except SecurityError as e:
            print(f"Security error: {e.message}")
        except TranslationError as e:
            print(f"Translation error: {e.message}")
    
    def example_batch_security_integration(self):
        """
        Example of how to integrate security with batch operations.
        """
        # This would be integrated into the batch manager
        texts = [
            "Hello world",
            "Contact me at john@example.com",
            "My SSN is 123-45-6789",
            "This is confidential information"  # Would be blocked if configured
        ]
        
        secure_texts = []
        all_pii_findings = []
        
        for i, text in enumerate(texts):
            try:
                sanitized, pii_findings = self.secure_service.security_manager.validate_and_sanitize_text(text)
                secure_texts.append(sanitized)
                all_pii_findings.extend(pii_findings)
                
                # Log each text processing
                self.secure_service.security_manager.log_operation(
                    operation="batch_text_processing",
                    text_length=len(text),
                    success=True
                )
                
            except (ValidationError, SecurityError) as e:
                print(f"Text {i} failed security check: {e.message}")
                # Log the failure
                self.secure_service.security_manager.log_operation(
                    operation="batch_text_processing",
                    text_length=len(text),
                    success=False,
                    error_code=e.error_code
                )
        
        print(f"Processed {len(secure_texts)} texts successfully")
        print(f"Total PII findings: {len(all_pii_findings)}")
        
        return secure_texts, all_pii_findings


if __name__ == "__main__":
    # Example usage
    example = SecurityIntegrationExample()
    
    print("=== Secure Translation Workflow ===")
    example.example_secure_translation_workflow()
    
    print("\n=== Batch Security Integration ===")
    example.example_batch_security_integration()