# Traigent SDK Disclaimer

This document provides cost, warranty, and operational-risk guidance for the Traigent SDK.
For paid services or portal users, contractual terms are governed by the applicable Traigent
Terms of Service, order form, or other signed agreement. For open-source SDK use without a
separate agreement, warranty and liability are governed by the AGPL-3.0 license; this document
is informational unless incorporated into a separate agreement.

## Cost Liability

Traigent optimizes LLM applications by running multiple trials across different configurations.
**This can result in significant API costs.**

### Cost Estimates Are Approximations

- Cost estimates provided by Traigent are **approximations based on maximum context length and expected token usage**
- **Actual billing is determined solely by the LLM provider** (OpenAI, Anthropic, Google, etc.)
- Traigent cannot guarantee cost accuracy for:
  - New or updated models with changed pricing
  - Custom endpoints or self-hosted models
  - Provider-specific billing variations
  - Token counting differences between providers

### User Responsibility for Costs

Users are **solely responsible** for:

1. Monitoring and controlling their API spending
2. Configuring appropriate spending limits via `TRAIGENT_RUN_COST_LIMIT`
3. Using `enable_mock_mode_for_quickstart()` from `traigent.testing` for local development and testing,
   or `python -m traigent.examples.quickstart` for the no-cost demo
4. Reviewing actual API provider billing independently
5. Understanding provider pricing before production use
6. Setting hard provider-side billing caps, quotas, budgets, and alerts directly with their LLM,
   API, cloud, and infrastructure providers

### Cost Limits Are Best-Effort Guardrails

Traigent cost limits, budgets, alerts, thresholds, stop conditions, and estimates are
**best-effort software controls**. They are not guarantees, insurance, financial commitments,
provider-side billing controls, or a substitute for provider/account-level spending caps.

Actual costs may exceed configured Traigent limits or estimates due to, among other things:

- Provider pricing changes, billing rules, or tokenization differences
- Parallel execution, retries, partial failures, timeouts, or cancelled requests
- Cached tokens, output tokens, tool calls, embeddings, or provider-specific usage accounting
- Stale pricing data, unrecognized model names, custom endpoints, or self-hosted models
- Bugs, race conditions, network failures, misconfiguration, or user-supplied credentials
- Provider dashboards or invoices applying charges differently than local estimates

### Parallel Execution Warning

Traigent may execute multiple LLM calls in parallel during optimization. This means:

- Multiple API requests can be made simultaneously
- Costs can accumulate rapidly during parallel execution
- Users should configure appropriate limits before running optimization

## Third-Party API Terms

Traigent integrates with third-party LLM providers. Users must comply with the terms of service
of each provider they use:

- **OpenAI**: https://openai.com/policies/terms-of-use
- **Anthropic**: https://www.anthropic.com/legal/consumer-terms
- **Google AI**: https://ai.google.dev/terms
- **AWS Bedrock**: https://aws.amazon.com/service-terms/
- **Azure OpenAI**: https://www.microsoft.com/licensing/terms/
- Other provider terms as applicable

Traigent is **not responsible** for:

- Provider billing, rate limits, or service interruptions
- Changes to provider pricing or terms
- Provider account suspension or termination
- Data handling by third-party providers
- API, cloud, infrastructure, or provider charges that exceed Traigent estimates, budgets,
  thresholds, alerts, stop conditions, or limits

## No Warranty

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.

This includes, but is not limited to:

- No warranty that cost estimates will be accurate
- No warranty that optimization will achieve desired results
- No warranty of uninterrupted or error-free operation
- No warranty regarding integration with third-party services

## Limitation of Liability

IN NO EVENT SHALL TRAIGENT, ITS AUTHORS, OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

**This specifically includes liability for:**

- API costs incurred during optimization
- Costs from misconfigured limits or settings
- Costs from parallel execution of LLM calls
- Costs from bugs, retries, race conditions, stale pricing data, custom model mappings, failed
  stop conditions, missed alerts, or provider billing differences
- Any financial damages related to LLM API usage

## Best Practices for Cost Control

To minimize unexpected costs, we strongly recommend:

1. **Always use mock mode for local development**:
   ```python
   from traigent.testing import enable_mock_mode_for_quickstart

   enable_mock_mode_for_quickstart()
   ```
   For the packaged demo, run `python -m traigent.examples.quickstart`. The legacy
   `TRAIGENT_MOCK_LLM=true` env var remains available for older local scripts and is disabled
   when `ENVIRONMENT=production`.

2. **Set explicit cost limits**:
   ```bash
   export TRAIGENT_RUN_COST_LIMIT=2.0  # USD per optimization run
   ```

3. **Start with small datasets and few trials**:
   ```python
   @traigent.optimize(max_trials=5)
   def my_function():
       ...
   ```

4. **Monitor your provider dashboards** for actual usage and billing

5. **Set hard caps directly with your providers** whenever available, including provider-side
   budgets, quotas, alerts, organization limits, project limits, and payment controls

6. **Use the approval flow** for production runs:
   ```bash
   export TRAIGENT_COST_APPROVED=true  # Only after reviewing estimates
   ```

## Customizing Cost Estimates

Traigent uses the [tokencost](https://github.com/AgentOps-AI/tokencost) library for LLM pricing data.

### Model Name Mappings

To add custom model name aliases, edit `traigent/utils/cost_calculator.py`:

```python
# In CostCalculator class
EXACT_MODEL_MAPPING = {
    # Add your custom mappings here
    "my-custom-model": "gpt-4o",  # Map to a known tokencost model
    "internal-claude": "claude-3-5-sonnet-20241022",
}
```

### Custom Pricing for Self-Hosted Models

For self-hosted or custom endpoint models not in tokencost:

1. The cost calculator will return `$0.00` for unknown models
2. Monitor actual costs via your infrastructure billing
3. Consider contributing new model pricing to the tokencost library

### Updating Pricing Data

To get the latest pricing data:

```bash
pip install --upgrade tokencost
```

Note: Pricing data in tokencost may lag behind provider announcements.

## Changes to This Disclaimer

This disclaimer may be updated from time to time. Continued use of the software constitutes
acceptance of any changes. Users are responsible for reviewing this disclaimer periodically.

## Contact

For questions about this disclaimer or the Traigent SDK, please open an issue on GitHub:
https://github.com/Traigent/Traigent/issues

---

*Last updated: May 2026*
*Licensed under GNU Affero General Public License v3.0 (AGPL-3.0-only) - See [LICENSE](LICENSE) for full terms*
