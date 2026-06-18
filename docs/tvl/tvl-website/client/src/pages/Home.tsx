import Layout from "@/components/Layout";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Link } from "wouter";
import { FileText, BookOpen, Code2, Zap, Shield, Layers } from "lucide-react";

export default function Home() {
  return (
    <Layout>
      {/* Hero Section */}
      <section className="relative py-20 md:py-32 overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-primary/10 via-transparent to-primary/5" />
        <div className="container relative">
          <div className="max-w-4xl mx-auto text-center space-y-8">
            <div className="inline-block px-4 py-2 rounded-full bg-primary/10 border border-primary/20 text-primary text-sm font-medium mb-4">
              Specification Backbone for AI Pipelines
            </div>
            <h1 className="text-5xl md:text-6xl lg:text-7xl font-bold tracking-tight">
              Tuned Variable Language
            </h1>
            <p className="text-xl md:text-2xl text-muted-foreground max-w-2xl mx-auto">
              TVL provides a robust specification framework for governed adaptation in AI pipelines, enabling software engineers to build reliable, maintainable, and adaptable AI systems.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center items-center pt-4">
              <Link href="/specification">
                <Button size="lg" className="text-base px-8">
                  <FileText className="mr-2 h-5 w-5" />
                  Read Specification
                </Button>
              </Link>
              <Link href="/book">
                <Button size="lg" variant="outline" className="text-base px-8">
                  <BookOpen className="mr-2 h-5 w-5" />
                  Read the Book
                </Button>
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 bg-card/50">
        <div className="container">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <Card className="border-border/50 bg-card/80 backdrop-blur">
              <CardHeader>
                <div className="w-12 h-12 rounded-lg bg-primary/10 flex items-center justify-center mb-4">
                  <Zap className="h-6 w-6 text-primary" />
                </div>
                <CardTitle>Governed Adaptation</CardTitle>
                <CardDescription>
                  Define and control how AI models adapt to changing requirements with precision and clarity.
                </CardDescription>
              </CardHeader>
            </Card>

            <Card className="border-border/50 bg-card/80 backdrop-blur">
              <CardHeader>
                <div className="w-12 h-12 rounded-lg bg-primary/10 flex items-center justify-center mb-4">
                  <Shield className="h-6 w-6 text-primary" />
                </div>
                <CardTitle>Type-Safe Specifications</CardTitle>
                <CardDescription>
                  Strong typing system ensures your AI pipeline configurations are validated at compile time.
                </CardDescription>
              </CardHeader>
            </Card>

            <Card className="border-border/50 bg-card/80 backdrop-blur">
              <CardHeader>
                <div className="w-12 h-12 rounded-lg bg-primary/10 flex items-center justify-center mb-4">
                  <Layers className="h-6 w-6 text-primary" />
                </div>
                <CardTitle>Pipeline Integration</CardTitle>
                <CardDescription>
                  Seamlessly integrate with existing AI pipelines and frameworks with minimal overhead.
                </CardDescription>
              </CardHeader>
            </Card>
          </div>
        </div>
      </section>

      {/* Resources Section */}
      <section className="py-20">
        <div className="container">
          <div className="text-center mb-12">
            <h2 className="text-3xl md:text-4xl font-bold mb-4">Explore TVL</h2>
            <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
              Comprehensive resources to help you master TVL and integrate it into your AI pipelines.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <Link href="/specification">
              <Card className="h-full border-border hover:border-primary/50 transition-all cursor-pointer group">
                <CardHeader>
                  <div className="w-12 h-12 rounded-lg bg-primary/10 flex items-center justify-center mb-4 group-hover:bg-primary/20 transition-colors">
                    <FileText className="h-6 w-6 text-primary" />
                  </div>
                  <CardTitle>Specification</CardTitle>
                  <CardDescription>
                    Explore the complete TVL language specification with syntax, semantics, and type system details.
                  </CardDescription>
                </CardHeader>
              </Card>
            </Link>

            <Link href="/book">
              <Card className="h-full border-border hover:border-primary/50 transition-all cursor-pointer group">
                <CardHeader>
                  <div className="w-12 h-12 rounded-lg bg-primary/10 flex items-center justify-center mb-4 group-hover:bg-primary/20 transition-colors">
                    <BookOpen className="h-6 w-6 text-primary" />
                  </div>
                  <CardTitle>The Book</CardTitle>
                  <CardDescription>
                    Learn TVL through comprehensive chapters covering motivation, constructs, and integration patterns.
                  </CardDescription>
                </CardHeader>
              </Card>
            </Link>

            <Link href="/examples">
              <Card className="h-full border-border hover:border-primary/50 transition-all cursor-pointer group">
                <CardHeader>
                  <div className="w-12 h-12 rounded-lg bg-primary/10 flex items-center justify-center mb-4 group-hover:bg-primary/20 transition-colors">
                    <Code2 className="h-6 w-6 text-primary" />
                  </div>
                  <CardTitle>Examples</CardTitle>
                  <CardDescription>
                    Get started quickly with real-world examples and project-ready patterns for your AI pipelines.
                  </CardDescription>
                </CardHeader>
              </Card>
            </Link>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 bg-gradient-to-br from-primary/10 via-primary/5 to-transparent">
        <div className="container">
          <div className="max-w-3xl mx-auto text-center space-y-6">
            <h2 className="text-3xl md:text-4xl font-bold">Ready to get started?</h2>
            <p className="text-lg text-muted-foreground">
              Whether you're a software engineer building production AI systems or a graduate student researching adaptive AI architectures, TVL provides the tools you need.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center items-center pt-4">
              <Link href="/examples">
                <Button size="lg" className="text-base px-8">
                  <Code2 className="mr-2 h-5 w-5" />
                  View Examples
                </Button>
              </Link>
            </div>
          </div>
        </div>
      </section>
    </Layout>
  );
}
