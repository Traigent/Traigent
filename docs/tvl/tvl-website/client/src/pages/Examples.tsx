import Layout from "@/components/Layout";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Download, Code2, FileCode, Copy, Check } from "lucide-react";
import { useState, useEffect } from "react";

interface ExampleFile {
  name: string;
  path: string;
  description: string;
  chapter?: string;
}

export default function Examples() {
  const [copiedFile, setCopiedFile] = useState<string | null>(null);
  const [fileContents, setFileContents] = useState<Record<string, string>>({});

  const examples: ExampleFile[] = [
    {
      name: "Motivation Experiment",
      path: "/examples/ch1_motivation_experiment.tvl.yml",
      description: "Chapter 1: Initial experiment demonstrating the need for governed adaptation",
      chapter: "1",
    },
    {
      name: "Hello TVL",
      path: "/examples/ch2_hello_tvl.tvl.yml",
      description: "Chapter 2: Basic TVL module with simple TVAR declarations",
      chapter: "2",
    },
    {
      name: "Validation Script",
      path: "/examples/ch2_validate_spec.py",
      description: "Chapter 2: Python script for validating TVL specifications",
      chapter: "2",
    },
    {
      name: "Constraints & Units",
      path: "/examples/ch3_constraints_units.tvl.yml",
      description: "Chapter 3: Structural constraints and domain-specific units",
      chapter: "3",
    },
    {
      name: "Constraint Tests",
      path: "/examples/ch3_constraint_tests.py",
      description: "Chapter 3: Testing framework for constraint validation",
      chapter: "3",
    },
    {
      name: "Environment Overlays",
      path: "/examples/ch4_environment_overlays.tvl.yml",
      description: "Chapter 4: Managing multiple deployment environments",
      chapter: "4",
    },
    {
      name: "Hotfix Overlay",
      path: "/examples/ch4_hotfix_overlay.tvl.yml",
      description: "Chapter 4: Emergency configuration overlay pattern",
      chapter: "4",
    },
    {
      name: "Integration Manifest",
      path: "/examples/ch5_integration_manifest.yaml",
      description: "Chapter 5: CI/CD integration manifest",
      chapter: "5",
    },
    {
      name: "Integration Pipeline",
      path: "/examples/ch5_integration_pipeline.sh",
      description: "Chapter 5: Shell script for pipeline integration",
      chapter: "5",
    },
    {
      name: "Promotion Policy",
      path: "/examples/promotion-policy.yml",
      description: "Complete promotion policy example with governance rules",
    },
  ];

  const loadFileContent = async (path: string) => {
    if (fileContents[path]) return fileContents[path];
    
    try {
      const response = await fetch(path);
      const content = await response.text();
      setFileContents((prev) => ({ ...prev, [path]: content }));
      return content;
    } catch (err) {
      console.error(`Error loading ${path}:`, err);
      return "Error loading file content";
    }
  };

  const copyToClipboard = async (path: string) => {
    const content = await loadFileContent(path);
    navigator.clipboard.writeText(content);
    setCopiedFile(path);
    setTimeout(() => setCopiedFile(null), 2000);
  };

  const groupedExamples = examples.reduce((acc, example) => {
    const chapter = example.chapter || "other";
    if (!acc[chapter]) acc[chapter] = [];
    acc[chapter].push(example);
    return acc;
  }, {} as Record<string, ExampleFile[]>);

  return (
    <Layout>
      {/* Header Section */}
      <section className="py-16 bg-gradient-to-br from-primary/10 via-transparent to-primary/5">
        <div className="container">
          <div className="max-w-4xl">
            <div className="flex items-center gap-3 mb-6">
              <Code2 className="h-12 w-12 text-primary" />
              <h1 className="text-4xl md:text-5xl font-bold">Examples</h1>
            </div>
            <p className="text-xl text-muted-foreground mb-8">
              Real-world TVL examples and project-ready patterns. Each example corresponds to chapters in the book 
              and demonstrates practical usage of TVL constructs.
            </p>
          </div>
        </div>
      </section>

      {/* Examples by Chapter */}
      <section className="py-16">
        <div className="container">
          <Tabs defaultValue="1" className="w-full">
            <TabsList className="grid w-full grid-cols-3 md:grid-cols-6 mb-8">
              <TabsTrigger value="1">Ch 1</TabsTrigger>
              <TabsTrigger value="2">Ch 2</TabsTrigger>
              <TabsTrigger value="3">Ch 3</TabsTrigger>
              <TabsTrigger value="4">Ch 4</TabsTrigger>
              <TabsTrigger value="5">Ch 5</TabsTrigger>
              <TabsTrigger value="other">Other</TabsTrigger>
            </TabsList>

            {Object.entries(groupedExamples).map(([chapter, chapterExamples]) => (
              <TabsContent key={chapter} value={chapter}>
                <div className="grid grid-cols-1 gap-6">
                  {chapterExamples.map((example) => (
                    <Card key={example.path}>
                      <CardHeader>
                        <div className="flex items-start justify-between">
                          <div className="flex-1">
                            <CardTitle className="flex items-center gap-2">
                              <FileCode className="h-5 w-5 text-primary" />
                              {example.name}
                            </CardTitle>
                            <CardDescription className="mt-2">{example.description}</CardDescription>
                          </div>
                          <div className="flex gap-2">
                            <Button
                              size="sm"
                              variant="outline"
                              onClick={() => copyToClipboard(example.path)}
                            >
                              {copiedFile === example.path ? (
                                <>
                                  <Check className="h-4 w-4 mr-1" />
                                  Copied
                                </>
                              ) : (
                                <>
                                  <Copy className="h-4 w-4 mr-1" />
                                  Copy
                                </>
                              )}
                            </Button>
                            <a href={example.path} download>
                              <Button size="sm">
                                <Download className="h-4 w-4 mr-1" />
                                Download
                              </Button>
                            </a>
                          </div>
                        </div>
                      </CardHeader>
                    </Card>
                  ))}
                </div>
              </TabsContent>
            ))}
          </Tabs>
        </div>
      </section>

      {/* Quick Start Section */}
      <section className="py-16 bg-card/50">
        <div className="container">
          <h2 className="text-3xl font-bold mb-8">Quick Start</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Validation Tooling</CardTitle>
                <CardDescription>
                  Use the TVL CLI tools to validate your specifications
                </CardDescription>
              </CardHeader>
              <CardContent>
                <pre className="text-sm">
                  <code>{`# Validate a TVL module
tvl-validate module.yml

# Lint for type errors
tvl-lint module.yml

# Check structural constraints
tvl-check-structural module.yml

# Validate configuration
tvl-config-validate config.yml`}</code>
                </pre>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Integration Example</CardTitle>
                <CardDescription>
                  Basic integration into your AI pipeline
                </CardDescription>
              </CardHeader>
              <CardContent>
                <pre className="text-sm">
                  <code>{`# Load TVL configuration
from tvl import load_module

# Parse module
module = load_module("agent.tvl.yml")

# Access tuned variables
config = module.get_config()
model = config.tvars["model"]
temperature = config.tvars["temperature"]`}</code>
                </pre>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* Use Cases Section */}
      <section className="py-16">
        <div className="container">
          <h2 className="text-3xl font-bold mb-8">Common Use Cases</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>RAG Support Bot</CardTitle>
                <CardDescription>
                  Enums, numeric domains, chance SLO, and DNF implications for retrieval-augmented generation
                </CardDescription>
              </CardHeader>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle>Text-to-SQL</CardTitle>
                <CardDescription>
                  Enumeration domains and banded targets for natural language database queries
                </CardDescription>
              </CardHeader>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle>Tool Use Agent</CardTitle>
                <CardDescription>
                  Fairness SLOs blocking promotions to ensure equitable AI agent behavior
                </CardDescription>
              </CardHeader>
            </Card>
          </div>
        </div>
      </section>
    </Layout>
  );
}
