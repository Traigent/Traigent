# Knowledge Graph Design for Optimizer Validation Test Analysis

## Problem Statement

The current test dashboard shows 379 tests but lacks:
1. **Navigation** - Hard to find related tests or understand coverage gaps
2. **Coverage Analysis** - No way to identify missing test combinations
3. **Semantic Queries** - Can't ask "which tests cover parallel execution with constraints?"
4. **Gap Detection** - Can't systematically identify untested combinations

## Solution: Test Knowledge Graph

Build an RDF knowledge graph encoding test dimensions, parameters, intents, and relationships.
This enables SPARQL queries for coverage analysis and visualization.

---

## Ontology Design

### Namespace Prefixes
```turtle
@prefix traigent: <http://traigent.ai/ontology/test#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .
```

### Core Classes

```turtle
# Test Dimensions (orthogonal axes of variation)
traigent:Dimension a rdfs:Class ;
    rdfs:label "Test Dimension" ;
    rdfs:comment "An orthogonal axis of variation in the optimizer" .

traigent:DimensionValue a rdfs:Class ;
    rdfs:label "Dimension Value" ;
    rdfs:comment "A specific value along a dimension" .

# Test Entities
traigent:TestFile a rdfs:Class ;
    rdfs:label "Test File" .

traigent:TestClass a rdfs:Class ;
    rdfs:label "Test Class" ;
    rdfs:comment "A pytest test class grouping related tests" .

traigent:TestCase a rdfs:Class ;
    rdfs:label "Test Case" ;
    rdfs:comment "An individual test method" .

# Configuration Space
traigent:ConfigParameter a rdfs:Class ;
    rdfs:label "Config Parameter" ;
    rdfs:comment "A parameter in the configuration space" .

traigent:ConfigSpace a rdfs:Class ;
    rdfs:label "Configuration Space" ;
    rdfs:comment "Complete configuration space for a test" .

# Expected Behavior
traigent:ExpectedOutcome a rdfs:Class ;
    rdfs:label "Expected Outcome" .

traigent:ValidationCheck a rdfs:Class ;
    rdfs:label "Validation Check" ;
    rdfs:comment "A specific assertion or validation performed" .

# Test Intent (what the test is trying to verify)
traigent:TestIntent a rdfs:Class ;
    rdfs:label "Test Intent" ;
    rdfs:comment "The semantic purpose of a test" .
```

### Properties

```turtle
# Dimension relationships
traigent:hasDimension a rdf:Property ;
    rdfs:domain traigent:TestCase ;
    rdfs:range traigent:Dimension .

traigent:hasDimensionValue a rdf:Property ;
    rdfs:domain traigent:TestCase ;
    rdfs:range traigent:DimensionValue .

traigent:belongsToDimension a rdf:Property ;
    rdfs:domain traigent:DimensionValue ;
    rdfs:range traigent:Dimension .

# Test structure
traigent:inFile a rdf:Property ;
    rdfs:domain traigent:TestClass ;
    rdfs:range traigent:TestFile .

traigent:inClass a rdf:Property ;
    rdfs:domain traigent:TestCase ;
    rdfs:range traigent:TestClass .

# Configuration
traigent:usesConfigSpace a rdf:Property ;
    rdfs:domain traigent:TestCase ;
    rdfs:range traigent:ConfigSpace .

traigent:hasParameter a rdf:Property ;
    rdfs:domain traigent:ConfigSpace ;
    rdfs:range traigent:ConfigParameter .

# Intent
traigent:verifiesIntent a rdf:Property ;
    rdfs:domain traigent:TestCase ;
    rdfs:range traigent:TestIntent .

# Outcomes
traigent:expectsOutcome a rdf:Property ;
    rdfs:domain traigent:TestCase ;
    rdfs:range traigent:ExpectedOutcome .

traigent:performsCheck a rdf:Property ;
    rdfs:domain traigent:TestCase ;
    rdfs:range traigent:ValidationCheck .

# Coverage relationships
traigent:coversInteraction a rdf:Property ;
    rdfs:domain traigent:TestCase ;
    rdfs:range traigent:DimensionValuePair ;
    rdfs:comment "Test covers interaction between two dimension values" .
```

---

## Dimension Taxonomy

Based on analysis of the test suite, here are the key dimensions:

### 1. Injection Mode
```turtle
traigent:InjectionMode a traigent:Dimension ;
    rdfs:label "Injection Mode" ;
    skos:definition "How config values are passed to the function" .

traigent:InjectionMode_Context a traigent:DimensionValue ;
    traigent:belongsToDimension traigent:InjectionMode ;
    rdfs:label "context" .

traigent:InjectionMode_Parameter a traigent:DimensionValue ;
    traigent:belongsToDimension traigent:InjectionMode ;
    rdfs:label "parameter" .

traigent:InjectionMode_Attribute a traigent:DimensionValue ;
    traigent:belongsToDimension traigent:InjectionMode ;
    rdfs:label "attribute" .

traigent:InjectionMode_Seamless a traigent:DimensionValue ;
    traigent:belongsToDimension traigent:InjectionMode ;
    rdfs:label "seamless" .
```

### 2. Execution Mode
```turtle
traigent:ExecutionMode a traigent:Dimension ;
    rdfs:label "Execution Mode" .

traigent:ExecutionMode_EdgeAnalytics a traigent:DimensionValue ;
    traigent:belongsToDimension traigent:ExecutionMode ;
    rdfs:label "edge_analytics" .

traigent:ExecutionMode_Privacy a traigent:DimensionValue ;
    traigent:belongsToDimension traigent:ExecutionMode ;
    rdfs:label "privacy" .

traigent:ExecutionMode_Hybrid a traigent:DimensionValue ;
    traigent:belongsToDimension traigent:ExecutionMode ;
    rdfs:label "hybrid" .

traigent:ExecutionMode_Cloud a traigent:DimensionValue ;
    traigent:belongsToDimension traigent:ExecutionMode ;
    rdfs:label "cloud" .
```

### 3. Optimizer Algorithm
```turtle
traigent:Algorithm a traigent:Dimension ;
    rdfs:label "Optimizer Algorithm" .

traigent:Algorithm_Random a traigent:DimensionValue ;
    traigent:belongsToDimension traigent:Algorithm ;
    rdfs:label "random" .

traigent:Algorithm_Grid a traigent:DimensionValue ;
    traigent:belongsToDimension traigent:Algorithm ;
    rdfs:label "grid" .

traigent:Algorithm_OptunaTPE a traigent:DimensionValue ;
    traigent:belongsToDimension traigent:Algorithm ;
    rdfs:label "optuna_tpe" .

traigent:Algorithm_OptunaCMAES a traigent:DimensionValue ;
    traigent:belongsToDimension traigent:Algorithm ;
    rdfs:label "optuna_cmaes" .
```

### 4. Config Space Type
```turtle
traigent:ConfigSpaceType a traigent:Dimension ;
    rdfs:label "Configuration Space Type" .

traigent:ConfigSpaceType_Categorical a traigent:DimensionValue ;
    traigent:belongsToDimension traigent:ConfigSpaceType ;
    rdfs:label "categorical_only" .

traigent:ConfigSpaceType_Continuous a traigent:DimensionValue ;
    traigent:belongsToDimension traigent:ConfigSpaceType ;
    rdfs:label "continuous_only" .

traigent:ConfigSpaceType_Mixed a traigent:DimensionValue ;
    traigent:belongsToDimension traigent:ConfigSpaceType ;
    rdfs:label "mixed" .
```

### 5. Objective Configuration
```turtle
traigent:ObjectiveConfig a traigent:Dimension ;
    rdfs:label "Objective Configuration" .

traigent:ObjectiveConfig_SingleMaximize a traigent:DimensionValue ;
    traigent:belongsToDimension traigent:ObjectiveConfig ;
    rdfs:label "single_maximize" .

traigent:ObjectiveConfig_SingleMinimize a traigent:DimensionValue ;
    traigent:belongsToDimension traigent:ObjectiveConfig ;
    rdfs:label "single_minimize" .

traigent:ObjectiveConfig_MultiObjective a traigent:DimensionValue ;
    traigent:belongsToDimension traigent:ObjectiveConfig ;
    rdfs:label "multi_objective" .

traigent:ObjectiveConfig_Weighted a traigent:DimensionValue ;
    traigent:belongsToDimension traigent:ObjectiveConfig ;
    rdfs:label "weighted" .
```

### 6. Stop Condition
```turtle
traigent:StopCondition a traigent:Dimension ;
    rdfs:label "Stop Condition" .

traigent:StopCondition_MaxTrials a traigent:DimensionValue ;
    traigent:belongsToDimension traigent:StopCondition ;
    rdfs:label "max_trials" .

traigent:StopCondition_Timeout a traigent:DimensionValue ;
    traigent:belongsToDimension traigent:StopCondition ;
    rdfs:label "timeout" .

traigent:StopCondition_Exhaustion a traigent:DimensionValue ;
    traigent:belongsToDimension traigent:StopCondition ;
    rdfs:label "config_exhaustion" .

traigent:StopCondition_Plateau a traigent:DimensionValue ;
    traigent:belongsToDimension traigent:StopCondition ;
    rdfs:label "plateau" .
```

### 7. Parallel Config
```turtle
traigent:ParallelMode a traigent:Dimension ;
    rdfs:label "Parallel Execution Mode" .

traigent:ParallelMode_Sequential a traigent:DimensionValue ;
    traigent:belongsToDimension traigent:ParallelMode ;
    rdfs:label "sequential" .

traigent:ParallelMode_Parallel a traigent:DimensionValue ;
    traigent:belongsToDimension traigent:ParallelMode ;
    rdfs:label "parallel" .

traigent:ParallelMode_Auto a traigent:DimensionValue ;
    traigent:belongsToDimension traigent:ParallelMode ;
    rdfs:label "auto" .
```

### 8. Constraint Usage
```turtle
traigent:ConstraintUsage a traigent:Dimension ;
    rdfs:label "Constraint Configuration" .

traigent:ConstraintUsage_None a traigent:DimensionValue ;
    traigent:belongsToDimension traigent:ConstraintUsage ;
    rdfs:label "no_constraints" .

traigent:ConstraintUsage_ConfigOnly a traigent:DimensionValue ;
    traigent:belongsToDimension traigent:ConstraintUsage ;
    rdfs:label "config_constraints" .

traigent:ConstraintUsage_MetricBased a traigent:DimensionValue ;
    traigent:belongsToDimension traigent:ConstraintUsage ;
    rdfs:label "metric_constraints" .

traigent:ConstraintUsage_Mixed a traigent:DimensionValue ;
    traigent:belongsToDimension traigent:ConstraintUsage ;
    rdfs:label "mixed_constraints" .
```

### 9. Failure Mode (for failure tests)
```turtle
traigent:FailureMode a traigent:Dimension ;
    rdfs:label "Expected Failure Mode" .

traigent:FailureMode_FunctionRaises a traigent:DimensionValue ;
    traigent:belongsToDimension traigent:FailureMode ;
    rdfs:label "function_raises" .

traigent:FailureMode_EvaluatorBug a traigent:DimensionValue ;
    traigent:belongsToDimension traigent:FailureMode ;
    rdfs:label "evaluator_bug" .

traigent:FailureMode_DatasetIssue a traigent:DimensionValue ;
    traigent:belongsToDimension traigent:FailureMode ;
    rdfs:label "dataset_issue" .

traigent:FailureMode_InvalidConfig a traigent:DimensionValue ;
    traigent:belongsToDimension traigent:FailureMode ;
    rdfs:label "invalid_config" .
```

---

## Test Intents Taxonomy

```turtle
# High-level intent categories
traigent:Intent_Correctness a traigent:TestIntent ;
    rdfs:label "Correctness" ;
    skos:definition "Verifies correct behavior under normal conditions" .

traigent:Intent_EdgeCase a traigent:TestIntent ;
    rdfs:label "Edge Case" ;
    skos:definition "Tests boundary conditions and edge cases" .

traigent:Intent_ErrorHandling a traigent:TestIntent ;
    rdfs:label "Error Handling" ;
    skos:definition "Verifies graceful error handling" .

traigent:Intent_Performance a traigent:TestIntent ;
    rdfs:label "Performance" ;
    skos:definition "Tests performance characteristics" .

traigent:Intent_Reproducibility a traigent:TestIntent ;
    rdfs:label "Reproducibility" ;
    skos:definition "Verifies deterministic/reproducible behavior" .

traigent:Intent_Integration a traigent:TestIntent ;
    rdfs:label "Integration" ;
    skos:definition "Tests interaction between components" .
```

---

## Example SPARQL Queries

### 1. Find Coverage Gaps (Missing Pairwise Combinations)

```sparql
PREFIX traigent: <http://traigent.ai/ontology/test#>

# Find all pairwise combinations of InjectionMode x ExecutionMode
# that are NOT covered by any test
SELECT ?injection ?execution
WHERE {
  # All injection modes
  ?injection traigent:belongsToDimension traigent:InjectionMode .
  # All execution modes
  ?execution traigent:belongsToDimension traigent:ExecutionMode .

  # Filter to those NOT covered
  FILTER NOT EXISTS {
    ?test a traigent:TestCase ;
          traigent:hasDimensionValue ?injection ;
          traigent:hasDimensionValue ?execution .
  }
}
```

### 2. Which Tests Cover Multi-Objective with Constraints?

```sparql
PREFIX traigent: <http://traigent.ai/ontology/test#>

SELECT ?test ?testClass ?file
WHERE {
  ?test a traigent:TestCase ;
        traigent:hasDimensionValue traigent:ObjectiveConfig_MultiObjective ;
        traigent:hasDimensionValue ?constraint ;
        traigent:inClass ?testClass .

  ?constraint traigent:belongsToDimension traigent:ConstraintUsage .
  FILTER (?constraint != traigent:ConstraintUsage_None)

  ?testClass traigent:inFile ?file .
}
```

### 3. Coverage Summary by Dimension

```sparql
PREFIX traigent: <http://traigent.ai/ontology/test#>

SELECT ?dimension ?value (COUNT(?test) AS ?testCount)
WHERE {
  ?test a traigent:TestCase ;
        traigent:hasDimensionValue ?value .
  ?value traigent:belongsToDimension ?dimension .
}
GROUP BY ?dimension ?value
ORDER BY ?dimension ?testCount
```

### 4. Find Tests with Multiple High-Risk Dimensions

```sparql
PREFIX traigent: <http://traigent.ai/ontology/test#>

# Tests that combine parallel execution + constraints + multi-objective
SELECT ?test ?testName
WHERE {
  ?test a traigent:TestCase ;
        rdfs:label ?testName ;
        traigent:hasDimensionValue ?parallel ;
        traigent:hasDimensionValue ?constraint ;
        traigent:hasDimensionValue ?objective .

  ?parallel traigent:belongsToDimension traigent:ParallelMode .
  FILTER (?parallel != traigent:ParallelMode_Sequential)

  ?constraint traigent:belongsToDimension traigent:ConstraintUsage .
  FILTER (?constraint != traigent:ConstraintUsage_None)

  ?objective traigent:belongsToDimension traigent:ObjectiveConfig .
  FILTER (?objective = traigent:ObjectiveConfig_MultiObjective)
}
```

### 5. Identify Untested Algorithm + Config Space Combinations

```sparql
PREFIX traigent: <http://traigent.ai/ontology/test#>

SELECT ?algorithm ?configType
WHERE {
  ?algorithm traigent:belongsToDimension traigent:Algorithm .
  ?configType traigent:belongsToDimension traigent:ConfigSpaceType .

  FILTER NOT EXISTS {
    ?test a traigent:TestCase ;
          traigent:hasDimensionValue ?algorithm ;
          traigent:hasDimensionValue ?configType .
  }
}
```

---

## Implementation Plan

### Phase 1: Graph Extraction (Python Script)

```python
# Extract test metadata to RDF
from rdflib import Graph, Namespace, Literal, URIRef
from rdflib.namespace import RDF, RDFS, XSD
import ast
import pytest

TRAIGENT = Namespace("http://traigent.ai/ontology/test#")

def extract_test_graph(test_dir: str) -> Graph:
    """Parse test files and extract to RDF graph."""
    g = Graph()
    g.bind("traigent", TRAIGENT)

    # Walk test files
    for test_file in Path(test_dir).glob("**/*.py"):
        extract_file_tests(g, test_file)

    return g

def infer_dimensions(scenario: dict) -> list[tuple]:
    """Infer dimension values from TestScenario attributes."""
    dimensions = []

    # Injection mode
    injection = scenario.get("injection_mode", "context")
    dimensions.append((TRAIGENT.InjectionMode,
                       TRAIGENT[f"InjectionMode_{injection.title()}"]))

    # Config space type
    config_space = scenario.get("config_space", {})
    config_type = analyze_config_space_type(config_space)
    dimensions.append((TRAIGENT.ConfigSpaceType,
                       TRAIGENT[f"ConfigSpaceType_{config_type}"]))

    # ... more dimension inference

    return dimensions
```

### Phase 2: Query Interface

Create a Python API for common queries:

```python
class TestKnowledgeGraph:
    def __init__(self, graph: Graph):
        self.g = graph

    def find_coverage_gaps(self, dim1: str, dim2: str) -> list[tuple]:
        """Find untested pairwise combinations."""
        query = f"""
        SELECT ?v1 ?v2
        WHERE {{
          ?v1 traigent:belongsToDimension traigent:{dim1} .
          ?v2 traigent:belongsToDimension traigent:{dim2} .
          FILTER NOT EXISTS {{
            ?test traigent:hasDimensionValue ?v1 ;
                  traigent:hasDimensionValue ?v2 .
          }}
        }}
        """
        return list(self.g.query(query))

    def suggest_new_tests(self) -> list[dict]:
        """Suggest new test cases to improve coverage."""
        gaps = []
        for d1, d2 in CRITICAL_DIMENSION_PAIRS:
            gaps.extend(self.find_coverage_gaps(d1, d2))
        return self._prioritize_gaps(gaps)
```

### Phase 3: Visualization

Integrate with the existing viewer:

```javascript
// Add to viewer/index.html
async function loadKnowledgeGraph() {
    const response = await fetch('/api/graph');
    const data = await response.json();

    // Build interactive dimension matrix
    renderCoverageMatrix(data.coverage);

    // Show gap analysis
    renderGapAnalysis(data.gaps);

    // Enable SPARQL query interface
    initQueryInterface(data.endpoint);
}

function renderCoverageMatrix(coverage) {
    // D3.js heatmap of dimension coverage
    const matrix = d3.select("#coverage-matrix")
        .append("svg")
        .attr("width", width)
        .attr("height", height);

    // Cells colored by coverage density
    // Click to filter tests by that combination
}
```

---

## SHACL Constraints for Test Validation

Define constraints that valid tests must satisfy:

```turtle
@prefix sh: <http://www.w3.org/ns/shacl#> .

# Every test must have at least one dimension value
traigent:TestCaseShape a sh:NodeShape ;
    sh:targetClass traigent:TestCase ;
    sh:property [
        sh:path traigent:hasDimensionValue ;
        sh:minCount 1 ;
        sh:message "Test must cover at least one dimension" ;
    ] ;
    sh:property [
        sh:path traigent:verifiesIntent ;
        sh:minCount 1 ;
        sh:message "Test must have a documented intent" ;
    ] .

# Config space must have valid parameters
traigent:ConfigSpaceShape a sh:NodeShape ;
    sh:targetClass traigent:ConfigSpace ;
    sh:property [
        sh:path traigent:hasParameter ;
        sh:minCount 1 ;
    ] .

# Failure tests must specify failure mode
traigent:FailureTestShape a sh:NodeShape ;
    sh:targetClass traigent:TestCase ;
    sh:property [
        sh:path traigent:expectsOutcome ;
        sh:hasValue traigent:Outcome_Failure ;
    ] ;
    sh:property [
        sh:path traigent:hasDimensionValue ;
        sh:qualifiedValueShape [
            sh:path traigent:belongsToDimension ;
            sh:hasValue traigent:FailureMode ;
        ] ;
        sh:qualifiedMinCount 1 ;
        sh:message "Failure tests must specify failure mode" ;
    ] .
```

---

## Coverage Metrics & Reporting

### Combinatorial Coverage Levels

| Level | Description | Target |
|-------|-------------|--------|
| 1-way | Each dimension value tested | 100% |
| 2-way | Each pair of values tested | 90% |
| 3-way | Each triple of values tested | 60% |
| Critical pairs | High-risk combinations | 100% |

### Critical Dimension Pairs (Must Test)

1. **InjectionMode × ExecutionMode** - Different injection flows in different modes
2. **Algorithm × ConfigSpaceType** - Algorithm behavior depends on space type
3. **ParallelMode × StopCondition** - Race conditions in parallel stopping
4. **ObjectiveConfig × ConstraintUsage** - Complex optimization scenarios
5. **Algorithm × FailureMode** - Error handling varies by algorithm

---

## Benefits

1. **Automated Gap Detection**: SPARQL queries identify untested combinations
2. **Semantic Navigation**: Find related tests by dimension or intent
3. **Coverage Visualization**: Interactive matrix showing test density
4. **Test Suggestions**: AI can suggest new tests to maximize coverage
5. **Regression Risk**: Identify tests affected by code changes via dimension overlap
6. **Documentation**: Graph serves as living documentation of test coverage

---

## References

- [GraphGen4Code](https://wala.github.io/graph4code/) - Code knowledge graph toolkit
- [SHACL W3C Spec](https://www.w3.org/TR/shacl/) - RDF validation constraints
- [Combinatorial Testing (NIST)](https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=910001) - Pairwise testing theory
- [Neo4j Codebase KG](https://neo4j.com/blog/developer/codebase-knowledge-graph/) - Code analysis with graphs
- [KG Test Case Generation](https://dl.acm.org/doi/abs/10.1145/3371158.3371202) - Academic paper on KG for test generation
