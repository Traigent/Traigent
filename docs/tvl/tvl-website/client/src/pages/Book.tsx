import Layout from "@/components/Layout";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Download, BookOpen, ArrowRight } from "lucide-react";
import { Link } from "wouter";
import { useState, useEffect } from "react";
import { Streamdown } from "streamdown";

export default function Book() {
  const [walkthroughsContent, setWalkthroughsContent] = useState("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch("/docs/walkthroughs.md")
      .then((r) => r.text())
      .then((content) => setWalkthroughsContent(content))
      .catch((err) => console.error("Error loading walkthroughs:", err))
      .finally(() => setLoading(false));
  }, []);

  const chapters = [
    {
      number: 1,
      title: "Motivation & Experiment Design",
      slug: "why-tvl-exists",
      description: "Understanding the need for governed adaptation in AI pipelines and experimental foundations.",
    },
    {
      number: 2,
      title: "Hello TVL",
      slug: "getting-fluent-in-tvl",
      description: "Getting started with TVL syntax, basic constructs, and validation workflows.",
    },
    {
      number: 3,
      title: "Constraints & Units",
      slug: "constraints-units-safety-nets",
      description: "Defining structural constraints, type systems, and domain-specific units for AI parameters.",
    },
    {
      number: 4,
      title: "Environment Overlays",
      slug: "patterns-for-real-deployments",
      description: "Managing multiple environments and applying configuration overlays for different deployment contexts.",
    },
    {
      number: 5,
      title: "Integration Patterns",
      slug: "integration-patterns",
      description: "Real-world integration pathways for incorporating TVL into existing AI pipelines and CI/CD workflows.",
    },
  ];

  return (
    <Layout>
      {/* Header Section */}
      <section className="py-16 bg-gradient-to-br from-primary/10 via-transparent to-primary/5">
        <div className="container">
          <div className="max-w-4xl">
            <div className="flex items-center gap-3 mb-6">
              <BookOpen className="h-12 w-12 text-primary" />
              <h1 className="text-4xl md:text-5xl font-bold">The TVL Book</h1>
            </div>
            <p className="text-xl text-muted-foreground mb-8">
              A comprehensive guide introducing the motivation, constructs, and integration pathways for TVL. 
              Designed for software engineers and graduate students working with AI pipelines.
            </p>
            <a href="/docs/specification.pdf" download>
              <Button size="lg">
                <Download className="mr-2 h-5 w-5" />
                Download Complete Book (PDF)
              </Button>
            </a>
          </div>
        </div>
      </section>

      {/* Chapters Section */}
      <section className="py-16">
        <div className="container">
          <h2 className="text-3xl font-bold mb-8">Chapters</h2>
          <div className="grid grid-cols-1 gap-6">
            {chapters.map((chapter) => (
              <Link key={chapter.number} href={`/book/chapter/${chapter.slug}`}>
                <Card className="hover:border-primary/50 hover:shadow-lg transition-all cursor-pointer group">
                  <CardHeader>
                    <div className="flex items-start gap-4">
                      <div className="flex-shrink-0 w-12 h-12 rounded-lg bg-primary/10 flex items-center justify-center group-hover:bg-primary/20 transition-colors">
                        <span className="text-xl font-bold text-primary">{chapter.number}</span>
                      </div>
                      <div className="flex-1">
                        <CardTitle className="text-xl mb-2 flex items-center justify-between">
                          {chapter.title}
                          <ArrowRight className="h-5 w-5 text-muted-foreground group-hover:text-primary group-hover:translate-x-1 transition-all" />
                        </CardTitle>
                        <CardDescription>{chapter.description}</CardDescription>
                      </div>
                    </div>
                  </CardHeader>
                </Card>
              </Link>
            ))}
          </div>
        </div>
      </section>

      {/* Walkthroughs Section */}
      <section className="py-16 bg-card/50">
        <div className="container">
          <h2 className="text-3xl font-bold mb-8">Walkthroughs</h2>
          <Card>
            <CardHeader>
              <CardTitle>Practical Examples & Tutorials</CardTitle>
              <CardDescription>
                Step-by-step guides through real-world TVL implementations
              </CardDescription>
            </CardHeader>
            <CardContent>
              {loading ? (
                <div className="text-center py-12 text-muted-foreground">Loading walkthroughs...</div>
              ) : (
                <div className="prose prose-invert max-w-none">
                  <Streamdown>{walkthroughsContent}</Streamdown>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </section>

      {/* Key Concepts Section */}
      <section className="py-16">
        <div className="container">
          <h2 className="text-3xl font-bold mb-8">Key Concepts</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Tuned Variables (TVARs)</CardTitle>
                <CardDescription>
                  Adaptive parameters in AI pipelines that are continuously optimized based on objectives and constraints, 
                  distinct from static configuration or runtime variables.
                </CardDescription>
              </CardHeader>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle>Structural Constraints</CardTitle>
                <CardDescription>
                  Type-checked logical expressions that define valid configuration spaces, ensuring AI systems 
                  operate within specified boundaries.
                </CardDescription>
              </CardHeader>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle>Objectives & SLOs</CardTitle>
                <CardDescription>
                  Quantifiable goals with target bands and chance constraints that guide the optimization 
                  process for AI pipeline parameters.
                </CardDescription>
              </CardHeader>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle>Promotion Policies</CardTitle>
                <CardDescription>
                  Governance rules that determine when and how configurations are promoted through 
                  different environments in your deployment pipeline.
                </CardDescription>
              </CardHeader>
            </Card>
          </div>
        </div>
      </section>
    </Layout>
  );
}
