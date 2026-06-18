import Layout from "@/components/Layout";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Download, FileText, Code } from "lucide-react";
import { useState, useEffect } from "react";
import { Streamdown } from "streamdown";

export default function Specification() {
  const [languageContent, setLanguageContent] = useState("");
  const [schemaContent, setSchemaContent] = useState("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      fetch("/docs/language.md").then((r) => r.text()),
      fetch("/docs/schema.md").then((r) => r.text()),
    ])
      .then(([lang, schema]) => {
        setLanguageContent(lang);
        setSchemaContent(schema);
      })
      .catch((err) => console.error("Error loading documentation:", err))
      .finally(() => setLoading(false));
  }, []);

  return (
    <Layout>
      {/* Header Section */}
      <section className="py-16 bg-gradient-to-br from-primary/10 via-transparent to-primary/5">
        <div className="container">
          <div className="max-w-4xl">
            <h1 className="text-4xl md:text-5xl font-bold mb-6">TVL Specification</h1>
            <p className="text-xl text-muted-foreground mb-8">
              Complete language reference including syntax, semantics, type system, and validation tooling for TVL.
            </p>
            <div className="flex flex-wrap gap-4">
              <a href="/docs/specification.pdf" download>
                <Button size="lg">
                  <Download className="mr-2 h-5 w-5" />
                  Download PDF
                </Button>
              </a>
              <a href="/schemas/tvl.schema.json" download>
                <Button size="lg" variant="outline">
                  <Code className="mr-2 h-5 w-5" />
                  JSON Schema
                </Button>
              </a>
              <a href="/schemas/tvl.ebnf" download>
                <Button size="lg" variant="outline">
                  <FileText className="mr-2 h-5 w-5" />
                  EBNF Grammar
                </Button>
              </a>
            </div>
          </div>
        </div>
      </section>

      {/* Documentation Content */}
      <section className="py-16">
        <div className="container">
          <Tabs defaultValue="language" className="w-full">
            <TabsList className="grid w-full max-w-md grid-cols-2">
              <TabsTrigger value="language">Language Reference</TabsTrigger>
              <TabsTrigger value="schema">Schema Reference</TabsTrigger>
            </TabsList>

            <TabsContent value="language" className="mt-8">
              <Card>
                <CardHeader>
                  <CardTitle>Language Reference</CardTitle>
                  <CardDescription>
                    Non-normative reference explaining TVL constructs with examples
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  {loading ? (
                    <div className="text-center py-12 text-muted-foreground">Loading documentation...</div>
                  ) : (
                    <div className="prose prose-invert max-w-none">
                      <Streamdown>{languageContent}</Streamdown>
                    </div>
                  )}
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="schema" className="mt-8">
              <Card>
                <CardHeader>
                  <CardTitle>Schema Reference</CardTitle>
                  <CardDescription>
                    Machine-readable schemas for TVL modules, configurations, and measurements
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  {loading ? (
                    <div className="text-center py-12 text-muted-foreground">Loading documentation...</div>
                  ) : (
                    <div className="prose prose-invert max-w-none">
                      <Streamdown>{schemaContent}</Streamdown>
                    </div>
                  )}
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>

          {/* Schema Downloads */}
          <div className="mt-12">
            <h2 className="text-2xl font-bold mb-6">Available Schemas</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              <a href="/schemas/tvl.schema.json" download>
                <Card className="h-full hover:border-primary/50 transition-all cursor-pointer">
                  <CardHeader>
                    <CardTitle className="text-base">TVL Module Schema</CardTitle>
                    <CardDescription className="text-sm">JSON Schema for TVL modules</CardDescription>
                  </CardHeader>
                </Card>
              </a>
              <a href="/schemas/tvl-configuration.schema.json" download>
                <Card className="h-full hover:border-primary/50 transition-all cursor-pointer">
                  <CardHeader>
                    <CardTitle className="text-base">Configuration Schema</CardTitle>
                    <CardDescription className="text-sm">Schema for configuration assignments</CardDescription>
                  </CardHeader>
                </Card>
              </a>
              <a href="/schemas/tvl-measurement.schema.json" download>
                <Card className="h-full hover:border-primary/50 transition-all cursor-pointer">
                  <CardHeader>
                    <CardTitle className="text-base">Measurement Schema</CardTitle>
                    <CardDescription className="text-sm">Schema for measurement bundles</CardDescription>
                  </CardHeader>
                </Card>
              </a>
              <a href="/schemas/tvl.ebnf" download>
                <Card className="h-full hover:border-primary/50 transition-all cursor-pointer">
                  <CardHeader>
                    <CardTitle className="text-base">EBNF Grammar</CardTitle>
                    <CardDescription className="text-sm">Extended Backus-Naur Form grammar</CardDescription>
                  </CardHeader>
                </Card>
              </a>
            </div>
          </div>
        </div>
      </section>
    </Layout>
  );
}
