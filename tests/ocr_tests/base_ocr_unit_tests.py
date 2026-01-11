"""
Base test class for OCR functionality across different providers.

This follows the same pattern as BaseLLMChatTest in tests/llm_translation/base_llm_unit_tests.py
"""
import pytest
import litellm
import os
from abc import ABC, abstractmethod
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider


# Test resources
TEST_IMAGE_PATH = "test_image_edit.png"
TEST_PDF_URL = "https://arxiv.org/pdf/2201.04234"


class BaseOCRTest(ABC):
    """
    Abstract base test class that enforces common OCR tests across all providers.
    
    Each provider-specific test class should inherit from this and implement
    get_base_ocr_call_args() to return provider-specific configuration.
    """

    @abstractmethod
    def get_base_ocr_call_args(self) -> dict:
        """Must return the base OCR call args for the specific provider"""
        pass

    @pytest.fixture(autouse=True)
    def _handle_rate_limits(self):
        """Fixture to handle rate limit errors for all test methods"""
        try:
            yield
        except litellm.RateLimitError as e:
            # Check if it's a quota exceeded error
            error_msg = str(e)
            if "Quota exceeded" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                pytest.skip(f"Quota exceeded - {error_msg}")
            else:
                pytest.skip(f"Rate limit exceeded - {error_msg}")
        except litellm.InternalServerError:
            pytest.skip("Model is overloaded")
        except litellm.BadRequestError as e:
            # Handle URL rejection errors from Vertex AI
            error_msg = str(e)
            if "URL_REJECTED" in error_msg or "Cannot fetch content from the provided URL" in error_msg:
                pytest.skip(f"URL rejected by provider - {error_msg}")
            else:
                raise

    @pytest.fixture(autouse=True)
    def _skip_if_missing_credentials(self):
        base_args = self.get_base_ocr_call_args()
        api_key = base_args.get("api_key")
        if api_key:
            return

        model = base_args.get("model") or ""
        custom_llm_provider = base_args.get("custom_llm_provider")
        api_base = base_args.get("api_base")

        _, provider, dynamic_api_key, _ = get_llm_provider(
            model=model,
            custom_llm_provider=custom_llm_provider,
            api_base=api_base,
            api_key=api_key,
        )

        if dynamic_api_key:
            return

        if provider == "mistral" and not os.getenv("MISTRAL_API_KEY"):
            pytest.skip("MISTRAL_API_KEY not set")
        if provider == "azure_ai":
            if not (os.getenv("AZURE_API_KEY") or os.getenv("AZURE_AI_API_KEY")):
                pytest.skip("AZURE_API_KEY not set")
            if not os.getenv("AZURE_API_BASE") and not api_base:
                pytest.skip("AZURE_API_BASE not set")
        if provider == "vertex_ai":
            has_creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            has_keypair = os.getenv("VERTEX_AI_PRIVATE_KEY") and os.getenv(
                "VERTEX_AI_PRIVATE_KEY_ID"
            )
            if not (has_creds or has_keypair):
                pytest.skip("Vertex AI credentials not set")

    @pytest.mark.parametrize("sync_mode", [True, False])
    @pytest.mark.asyncio
    async def test_basic_ocr_with_url(self, sync_mode):
        """
        Test basic OCR with a public URL.
        """
        litellm._turn_on_debug()
        base_ocr_call_args = self.get_base_ocr_call_args()
        print("BASE OCR Call args=", base_ocr_call_args)
        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        litellm.model_cost = litellm.get_model_cost_map(url="")

        try:
            if sync_mode:
                response = litellm.ocr(
                    document={
                        "type": "document_url",
                        "document_url": TEST_PDF_URL
                    },
                    **base_ocr_call_args,
                )
            else:
                response = await litellm.aocr(
                    document={
                        "type": "document_url",
                        "document_url": TEST_PDF_URL
                    },
                    **base_ocr_call_args,
                )

            print(f"\n{'='*80}")
            print(f"Sync Mode: {sync_mode}")
            print(f"Response type: {type(response)}")
            print(f"Response object: {response.object if hasattr(response, 'object') else 'N/A'}")
            
            # Check if response has expected OCR format
            assert hasattr(response, "pages"), "Response should have 'pages' attribute"
            assert hasattr(response, "model"), "Response should have 'model' attribute"
            assert hasattr(response, "object"), "Response should have 'object' attribute"
            assert response.object == "ocr", f"Expected object='ocr', got '{response.object}'"
            
            # Validate pages structure
            assert isinstance(response.pages, list), "pages should be a list"
            assert len(response.pages) > 0, "Should have at least one page"
            
            # Check first page structure
            first_page = response.pages[0]
            assert hasattr(first_page, "index"), "Page should have 'index' attribute"
            assert hasattr(first_page, "markdown"), "Page should have 'markdown' attribute"
            
            # Extract text from all pages for validation
            total_text = "\n\n".join(page.markdown for page in response.pages if page.markdown)
            print(f"Total pages: {len(response.pages)}")
            print(f"Total extracted text length: {len(total_text)} characters")
            print(f"First 200 chars: {total_text[:200]}")
            print(f"Model: {response.model}")
            if response.usage_info:
                print(f"Pages processed: {response.usage_info.pages_processed}")
            print(f"{'='*80}\n")
            
            assert len(total_text) > 0, "Should extract some text from the document"

            #########################################################
            # validate we get a response cost in hidden parameters
            #########################################################
            hidden_params = response._hidden_params
            assert isinstance(hidden_params, dict), "Hidden parameters should be a dictionary"

            print("response usage_info:", response.usage_info)

            response_cost = hidden_params.get("response_cost")
            assert response_cost is not None, "Response cost should be in hidden parameters"
            assert response_cost > 0, "Response cost should be greater than 0"
            print("response_cost=", response_cost)
            
        except (litellm.RateLimitError, litellm.InternalServerError):
            # Re-raise these errors so the fixture can handle them
            raise
        except Exception as e:
            pytest.fail(f"OCR call failed: {str(e)}")

    def test_ocr_response_structure(self):
        """
        Test that the OCR response has the correct structure.
        """
        litellm.set_verbose = True
        base_ocr_call_args = self.get_base_ocr_call_args()

        try:
            response = litellm.ocr(
                document={
                    "type": "document_url",
                    "document_url": TEST_PDF_URL
                },
                **base_ocr_call_args,
            )

            # Validate response structure
            assert hasattr(response, "pages"), "Response should have 'pages' attribute"
            assert hasattr(response, "model"), "Response should have 'model' attribute"
            assert hasattr(response, "object"), "Response should have 'object' attribute"
            assert hasattr(response, "usage_info"), "Response should have 'usage_info' attribute"
            
            assert isinstance(response.pages, list), "pages should be a list"
            assert len(response.pages) > 0, "Should have at least one page"
            assert response.object == "ocr", "object should be 'ocr'"
            
            # Validate first page structure
            first_page = response.pages[0]
            assert hasattr(first_page, "index"), "Page should have 'index' attribute"
            assert hasattr(first_page, "markdown"), "Page should have 'markdown' attribute"
            assert isinstance(first_page.markdown, str), "markdown should be a string"
            
            print(f"\nResponse structure validated:")
            print(f"  - object: {response.object}")
            print(f"  - model: {response.model}")
            print(f"  - pages: {len(response.pages)}")
            if response.usage_info:
                print(f"  - pages_processed: {response.usage_info.pages_processed}")
                print(f"  - doc_size_bytes: {response.usage_info.doc_size_bytes}")
        
        except (litellm.RateLimitError, litellm.InternalServerError):
            # Re-raise these errors so the fixture can handle them
            raise
        except Exception as e:
            pytest.fail(f"OCR response structure test failed: {str(e)}")
