import Layout from "@/components/Layout";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { useRoute, Link } from "wouter";
import { ArrowLeft, Copy, Download, Check } from "lucide-react";
import { useState, useEffect } from "react";
import { toast } from "sonner";

interface SectionContent {
  title: string;
  description: string;
  content: string[];
  codeExample?: {
    language: string;
    filename: string;
    code: string;
    downloadPath: string;
  };
  additionalExample?: {
    language: string;
    filename: string;
    code: string;
    downloadPath: string;
  };
}

interface SectionsData {
  [key: string]: SectionContent;
}

export default function Section() {
  const [, params] = useRoute("/book/chapter/:chapterSlug/section/:sectionSlug");
  const [section, setSection] = useState<SectionContent | null>(null);
  const [loading, setLoading] = useState(true);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    const sectionKey = `${params?.chapterSlug}-${params?.sectionSlug}`;
    fetch("/docs/sections.json")
      .then((r) => r.json())
      .then((data: SectionsData) => {
        setSection(data[sectionKey] || null);
      })
      .catch((err) => console.error("Error loading section:", err))
      .finally(() => setLoading(false));
  }, [params?.chapterSlug, params?.sectionSlug]);

  const [copiedAdditional, setCopiedAdditional] = useState(false);

  const handleCopy = () => {
    if (section?.codeExample) {
      navigator.clipboard.writeText(section.codeExample.code);
      setCopied(true);
      toast.success("Code copied to clipboard!");
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const handleCopyAdditional = () => {
    if (section?.additionalExample) {
      navigator.clipboard.writeText(section.additionalExample.code);
      setCopiedAdditional(true);
      toast.success("Code copied to clipboard!");
      setTimeout(() => setCopiedAdditional(false), 2000);
    }
  };

  if (loading) {
    return (
      <Layout>
        <div className="container py-16">
          <div className="text-center text-muted-foreground">Loading section...</div>
        </div>
      </Layout>
    );
  }

  if (!section) {
    return (
      <Layout>
        <div className="container py-16">
          <div className="text-center">
            <h1 className="text-3xl font-bold mb-4">Section Not Found</h1>
            <Link href={`/book/chapter/${params?.chapterSlug}`}>
              <Button>
                <ArrowLeft className="mr-2 h-4 w-4" />
                Back to Chapter
              </Button>
            </Link>
          </div>
        </div>
      </Layout>
    );
  }

  return (
    <Layout>
      {/* Header Section */}
      <section className="py-16 bg-gradient-to-br from-primary/10 via-transparent to-primary/5">
        <div className="container">
          <Link href={`/book/chapter/${params?.chapterSlug}`}>
            <Button variant="ghost" className="mb-6">
              <ArrowLeft className="mr-2 h-4 w-4" />
              Back to Chapter
            </Button>
          </Link>
          <div className="max-w-4xl">
            <h1 className="text-4xl md:text-5xl font-bold mb-4">{section.title}</h1>
            <p className="text-xl text-muted-foreground">{section.description}</p>
          </div>
        </div>
      </section>

      {/* Content Section */}
      <section className="py-16">
        <div className="container max-w-4xl">
          <div className="space-y-6">
            {section.content.map((paragraph, index) => (
              <p key={index} className="text-lg leading-relaxed text-muted-foreground">
                {paragraph}
              </p>
            ))}
          </div>

          {/* Code Example */}
          {section.codeExample && (
            <Card className="mt-12">
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle className="text-lg font-mono">{section.codeExample.filename}</CardTitle>
                    <CardDescription className="mt-1">
                      {section.codeExample.language === "yaml" ? "TVL Specification" : "Python Validation Script"}
                    </CardDescription>
                  </div>
                  <div className="flex gap-2">
                    <Button variant="outline" size="sm" onClick={handleCopy}>
                      {copied ? (
                        <>
                          <Check className="mr-2 h-4 w-4" />
                          Copied!
                        </>
                      ) : (
                        <>
                          <Copy className="mr-2 h-4 w-4" />
                          Copy
                        </>
                      )}
                    </Button>
                    <a href={section.codeExample.downloadPath} download>
                      <Button variant="outline" size="sm">
                        <Download className="mr-2 h-4 w-4" />
                        Download
                      </Button>
                    </a>
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                <pre className="bg-muted/50 p-4 rounded-lg overflow-x-auto">
                  <code className="text-sm font-mono">{section.codeExample.code}</code>
                </pre>
              </CardContent>
            </Card>
          )}

          {/* Additional Code Example */}
          {section.additionalExample && (
            <Card className="mt-8">
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle className="text-lg font-mono">{section.additionalExample.filename}</CardTitle>
                    <CardDescription className="mt-1">
                      {section.additionalExample.language === "bash" ? "Integration Pipeline Script" : "Additional Example"}
                    </CardDescription>
                  </div>
                  <div className="flex gap-2">
                    <Button variant="outline" size="sm" onClick={handleCopyAdditional}>
                      {copiedAdditional ? (
                        <>
                          <Check className="mr-2 h-4 w-4" />
                          Copied!
                        </>
                      ) : (
                        <>
                          <Copy className="mr-2 h-4 w-4" />
                          Copy
                        </>
                      )}
                    </Button>
                    <a href={section.additionalExample.downloadPath} download>
                      <Button variant="outline" size="sm">
                        <Download className="mr-2 h-4 w-4" />
                        Download
                      </Button>
                    </a>
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                <pre className="bg-muted/50 p-4 rounded-lg overflow-x-auto">
                  <code className="text-sm font-mono">{section.additionalExample.code}</code>
                </pre>
              </CardContent>
            </Card>
          )}

          {/* Navigation */}
          <div className="flex justify-between items-center mt-12 pt-8 border-t border-border">
            <Link href={`/book/chapter/${params?.chapterSlug}`}>
              <Button variant="outline">
                <ArrowLeft className="mr-2 h-4 w-4" />
                Back to Chapter
              </Button>
            </Link>
          </div>
        </div>
      </section>
    </Layout>
  );
}
