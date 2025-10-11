from typing import Any, Dict, List, Optional
from logging import Logger, getLogger

from pydantic import BaseModel

from datasets import load_dataset

from src.data.env import EnvDataset, EnvDatasetConfig, Environment
from src.data.envs.qa_env import QAEnv


def _normalize_text(s: str) -> str:
    return " ".join((s or "").strip().split())


class FinerEnv(QAEnv):
    """Single-turn QA environment for FinLoRA Finer task.

    Expects the model to output exactly the US GAAP tag string.
    """

    def evaluate(self, action: str) -> Dict[str, Any]:
        predicted_answer = action
        # Reuse QAEnv's boxed extraction behavior
        if "\\boxed{" in (predicted_answer or ""):
            try:
                predicted_answer = predicted_answer.split("\\boxed{")[1].split("}")[0]
            except Exception:
                pass
        pred_norm = _normalize_text(predicted_answer)
        tgt_norm = _normalize_text(self.answer)
        correct = pred_norm == tgt_norm
        message = "Correct!" if correct else f"Incorrect! The correct answer is {self.answer}."
        return {"score": 1 if correct else 0, "target": self.answer, "message": message}


class FinerEnvDatasetConfig(EnvDatasetConfig):
    """Config for loading FinLoRA Finer datasets.

    Supports either a local JSONL path (dataset_path) or a Hugging Face dataset
    specified by hf_dataset[/hf_subset] and split.
    """

    # Local JSONL (one JSON object per line)
    dataset_path: Optional[str] = None

    # Hugging Face dataset identifiers
    hf_dataset: Optional[str] = None
    hf_subset: Optional[str] = None
    split: str = "test"

    # Field mapping
    input_field: str = "context"
    target_field: str = "target"
    instruction_template: Optional[str] = None

    # Sampling/shuffle
    max_samples: Optional[int] = None
    shuffle: bool = False
    seed: int = 42
    
    # Options
    prepend_options: bool = False


class FinerEnvDataset(EnvDataset):
    def __init__(self, config: FinerEnvDatasetConfig, logger: Optional[Logger] = None):
        self.config = config
        self.logger = logger or getLogger("finer_dataset")
        self.dataset = self.load_dataset()

    def _build_env(self, question: str, answer: str, metadata: Dict[str, Any]) -> Environment:
        
        if self.config.prepend_options:
            question = f"Here is a list of US GAAP tags options: {', '.join(OPTIONS)}.\n\n-> {question}"
            
        return FinerEnv(
            env_id=metadata.get("id", "finer"),
            env_type=metadata.get("task", "finer"),
            question=question,
            answer=answer,
            metadata=metadata,
            instruction_template=self.config.instruction_template,
            logger=self.logger,
        )

    def _load_local_jsonl(self, path: str) -> List[Environment]:
        import json
        envs: List[Environment] = []
        in_key = self.config.input_field
        tgt_key = self.config.target_field
        count = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ex = json.loads(line)
                except Exception:
                    continue
                q = ex.get(in_key)
                a = ex.get(tgt_key)
                if not isinstance(q, str) or not isinstance(a, str):
                    continue
                metadata = {k: v for k, v in ex.items() if k not in (in_key, tgt_key)}
                envs.append(self._build_env(question=q, answer=a, metadata=metadata))
                count += 1
                if self.config.max_samples is not None and count >= self.config.max_samples:
                    break
        return envs

    def _load_hf(self) -> List[Environment]:
        if self.config.hf_subset:
            ds = load_dataset(self.config.hf_dataset, self.config.hf_subset)
        else:
            ds = load_dataset(self.config.hf_dataset)
        if self.config.split not in ds:
            raise ValueError(
                f"Split '{self.config.split}' not found. Available: {list(ds.keys())}"
            )
        split_ds = ds[self.config.split]
        if self.config.shuffle:
            split_ds = split_ds.shuffle(seed=self.config.seed)
        if self.config.max_samples is not None:
            split_ds = split_ds.select(range(min(self.config.max_samples, len(split_ds))))

        envs: List[Environment] = []
        in_key = self.config.input_field
        tgt_key = self.config.target_field
        for ex in split_ds:
            q = ex.get(in_key)
            a = ex.get(tgt_key)
            if not isinstance(q, str) or not isinstance(a, str):
                continue
            metadata = {k: v for k, v in ex.items() if k not in (in_key, tgt_key)}
            envs.append(self._build_env(question=q, answer=a, metadata=metadata))
        return envs

    def load_dataset(self) -> List[Environment]:
        # Prefer local file when provided
        if isinstance(self.config.dataset_path, str) and len(self.config.dataset_path.strip()) > 0:
            return self._load_local_jsonl(self.config.dataset_path)

        # Fallback to HF if specified
        if isinstance(self.config.hf_dataset, str) and len(self.config.hf_dataset.strip()) > 0:
            return self._load_hf()

        # Nothing specified
        raise ValueError(
            "FinerEnvDataset requires either dataset_path (local JSONL) or hf_dataset (Hugging Face)."
        )


OPTIONS = ['SharebasedCompensationArrangementBySharebasedPaymentAwardAwardVestingRightsPercentage',
 'InterestExpense',
 'GoodwillImpairmentLoss',
 'SaleOfStockPricePerShare',
 'BusinessCombinationAcquisitionRelatedCosts',
 'LineOfCreditFacilityCurrentBorrowingCapacity',
 'LineOfCreditFacilityMaximumBorrowingCapacity',
 'PreferredStockSharesAuthorized',
 'RestructuringCharges',
 'IncomeLossFromEquityMethodInvestments',
 'EquityMethodInvestmentOwnershipPercentage',
 'Revenues',
 'NumberOfRealEstateProperties',
 'CumulativeEffectOfNewAccountingPrincipleInPeriodOfAdoption',
 'IncomeTaxExpenseBenefit',
 'SharebasedCompensationArrangementBySharebasedPaymentAwardExpirationPeriod',
 'DebtInstrumentFairValue',
 'AccrualForEnvironmentalLossContingencies',
 'CommonStockDividendsPerShareDeclared',
 'UnrecognizedTaxBenefitsThatWouldImpactEffectiveTaxRate',
 'Goodwill',
 'CommonStockSharesAuthorized',
 'UnrecognizedTaxBenefits',
 'LineOfCredit',
 'PublicUtilitiesRequestedRateIncreaseDecreaseAmount',
 'EquityMethodInvestments',
 'LineOfCreditFacilityUnusedCapacityCommitmentFeePercentage',
 'ShareBasedCompensationArrangementByShareBasedPaymentAwardEquityInstrumentsOtherThanOptionsVestedInPeriodTotalFairValue',
 'CommonStockCapitalSharesReservedForFutureIssuance',
 'DebtInstrumentConvertibleConversionPrice1',
 'LossContingencyPendingClaimsNumber',
 'OperatingLeasePayments',
 'LongTermDebtFairValue',
 'LeaseAndRentalExpense',
 'OperatingLeaseWeightedAverageRemainingLeaseTerm1',
 'LongTermDebt',
 'ClassOfWarrantOrRightExercisePriceOfWarrantsOrRights1',
 'DefinedContributionPlanCostRecognized',
 'LesseeOperatingLeaseTermOfContract',
 'ShareBasedCompensationArrangementByShareBasedPaymentAwardNumberOfSharesAuthorized',
 'DebtWeightedAverageInterestRate',
 'GuaranteeObligationsMaximumExposure',
 'DebtInstrumentTerm',
 'CapitalizedContractCostAmortization',
 'FiniteLivedIntangibleAssetUsefulLife',
 'ShareBasedCompensationArrangementByShareBasedPaymentAwardOptionsGrantsInPeriodGross',
 'DebtInstrumentInterestRateEffectivePercentage',
 'LettersOfCreditOutstandingAmount',
 'NumberOfOperatingSegments',
 'AllocatedShareBasedCompensationExpense',
 'CashAndCashEquivalentsFairValueDisclosure',
 'ContractWithCustomerLiabilityRevenueRecognized',
 'EmployeeServiceShareBasedCompensationTaxBenefitFromCompensationExpense',
 'LineOfCreditFacilityCommitmentFeePercentage',
 'DerivativeNotionalAmount',
 'AntidilutiveSecuritiesExcludedFromComputationOfEarningsPerShareAmount',
 'TreasuryStockAcquiredAverageCostPerShare',
 'RevenueFromRelatedParties',
 'BusinessAcquisitionPercentageOfVotingInterestsAcquired',
 'AmortizationOfIntangibleAssets',
 'BusinessCombinationRecognizedIdentifiableAssetsAcquiredAndLiabilitiesAssumedIntangibleAssetsOtherThanGoodwill',
 'ContractWithCustomerLiability',
 'ShareBasedCompensationArrangementByShareBasedPaymentAwardOptionsGrantsInPeriodWeightedAverageGrantDateFairValue',
 'AssetImpairmentCharges',
 'DebtInstrumentBasisSpreadOnVariableRate1',
 'BusinessCombinationConsiderationTransferred1',
 'DebtInstrumentUnamortizedDiscount',
 'PaymentsToAcquireBusinessesNetOfCashAcquired',
 'ShareBasedCompensationArrangementByShareBasedPaymentAwardAwardVestingPeriod1',
 'DebtInstrumentCarryingAmount',
 'AcquiredFiniteLivedIntangibleAssetsWeightedAverageUsefulLife',
 'DerivativeFixedInterestRate',
 'ShareBasedCompensationArrangementByShareBasedPaymentAwardEquityInstrumentsOtherThanOptionsGrantsInPeriod',
 'TreasuryStockValueAcquiredCostMethod',
 'OperatingLossCarryforwards',
 'DebtInstrumentMaturityDate',
 'ShareBasedCompensationArrangementByShareBasedPaymentAwardEquityInstrumentsOtherThanOptionsNonvestedNumber',
 'DefinedBenefitPlanContributionsByEmployer',
 'GainsLossesOnExtinguishmentOfDebt',
 'AreaOfRealEstateProperty',
 'BusinessAcquisitionEquityInterestsIssuedOrIssuableNumberOfSharesIssued',
 'SaleOfStockNumberOfSharesIssuedInTransaction',
 'SupplementalInformationForPropertyCasualtyInsuranceUnderwritersPriorYearClaimsAndClaimsAdjustmentExpense',
 'RevenueFromContractWithCustomerIncludingAssessedTax',
 'DeferredFinanceCostsGross',
 'NumberOfReportableSegments',
 'BusinessCombinationContingentConsiderationLiability',
 'ShareBasedCompensationArrangementByShareBasedPaymentAwardEquityInstrumentsOtherThanOptionsGrantsInPeriodWeightedAverageGrantDateFairValue',
 'RepaymentsOfDebt',
 'SharePrice',
 'ShareBasedCompensationArrangementByShareBasedPaymentAwardNumberOfSharesAvailableForGrant',
 'StockRepurchaseProgramAuthorizedAmount1',
 'LineOfCreditFacilityRemainingBorrowingCapacity',
 'PropertyPlantAndEquipmentUsefulLife',
 'ShareBasedCompensationArrangementByShareBasedPaymentAwardOptionsExercisesInPeriodTotalIntrinsicValue',
 'DisposalGroupIncludingDiscontinuedOperationConsideration',
 'DebtInstrumentRedemptionPricePercentage',
 'DebtInstrumentInterestRateStatedPercentage',
 'OperatingLeasesRentExpenseNet',
 'StockRepurchaseProgramRemainingAuthorizedRepurchaseAmount1',
 'AmortizationOfFinancingCosts',
 'ConcentrationRiskPercentage1',
 'Depreciation',
 'RevenueFromContractWithCustomerExcludingAssessedTax',
 'RelatedPartyTransactionExpensesFromTransactionsWithRelatedParty',
 'DebtInstrumentFaceAmount',
 'RestructuringAndRelatedCostExpectedCost1',
 'EmployeeServiceShareBasedCompensationNonvestedAwardsTotalCompensationCostNotYetRecognizedPeriodForRecognition1',
 'MinorityInterestOwnershipPercentageByNoncontrollingOwners',
 'CommonStockParOrStatedValuePerShare',
 'MinorityInterestOwnershipPercentageByParent',
 'CommonStockSharesOutstanding',
 'EmployeeServiceShareBasedCompensationNonvestedAwardsTotalCompensationCostNotYetRecognized',
 'DeferredFinanceCostsNet',
 'ShareBasedCompensation',
 'InterestExpenseDebt',
 'StockIssuedDuringPeriodSharesNewIssues',
 'EffectiveIncomeTaxRateContinuingOperations',
 'BusinessCombinationRecognizedIdentifiableAssetsAcquiredAndLiabilitiesAssumedIntangibles',
 'OperatingLeaseExpense',
 'PreferredStockDividendRatePercentage',
 'StockRepurchasedDuringPeriodShares',
 'OperatingLeaseCost',
 'ProceedsFromIssuanceOfCommonStock',
 'StockRepurchasedAndRetiredDuringPeriodShares',
 'RelatedPartyTransactionAmountsOfTransaction',
 'EmployeeServiceShareBasedCompensationNonvestedAwardsTotalCompensationCostNotYetRecognizedShareBasedAwardsOtherThanOptions',
 'OperatingLeaseLiability',
 'EffectiveIncomeTaxRateReconciliationAtFederalStatutoryIncomeTaxRate',
 'OperatingLeaseWeightedAverageDiscountRatePercent',
 'PaymentsToAcquireBusinessesGross',
 'LossContingencyDamagesSoughtValue',
 'TreasuryStockSharesAcquired',
 'LossContingencyAccrualAtCarryingValue',
 'RevenueRemainingPerformanceObligation',
 'LineOfCreditFacilityInterestRateAtPeriodEnd',
 'LesseeOperatingLeaseRenewalTerm',
 'OperatingLeaseRightOfUseAsset',
 'LossContingencyEstimateOfPossibleLoss']