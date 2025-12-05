import Layout from "@/components/Layout";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { useRoute, Link } from "wouter";
import { ArrowLeft, ArrowRight, BookOpen, Download, ChevronRight } from "lucide-react";
import { useState, useEffect } from "react";

interface ChapterSection {
  title: string;
  content: string;
}

interface Chapter {
  id: number;
  title: string;
  slug: string;
  description: string;
  sections: ChapterSection[];
}

interface ChaptersData {
  chapters: Chapter[];
}

export default function Chapter() {
  const [, params] = useRoute("/book/chapter/:slug");
  const [chapter, setChapter] = useState<Chapter | null>(null);
  const [allChapters, setAllChapters] = useState<Chapter[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch("/docs/chapters.json")
      .then((r) => r.json())
      .then((data: ChaptersData) => {
        setAllChapters(data.chapters);
        const found = data.chapters.find((c) => c.slug === params?.slug);
        setChapter(found || null);
      })
      .catch((err) => console.error("Error loading chapter:", err))
      .finally(() => setLoading(false));
  }, [params?.slug]);

  if (loading) {
    return (
      <Layout>
        <div className="container py-16">
          <div className="text-center text-muted-foreground">Loading chapter...</div>
        </div>
      </Layout>
    );
  }

  if (!chapter) {
    return (
      <Layout>
        <div className="container py-16">
          <div className="text-center">
            <h1 className="text-3xl font-bold mb-4">Chapter Not Found</h1>
            <Link href="/book">
              <Button>
                <ArrowLeft className="mr-2 h-4 w-4" />
                Back to Book
              </Button>
            </Link>
          </div>
        </div>
      </Layout>
    );
  }

  const prevChapter = allChapters.find((c) => c.id === chapter.id - 1);
  const nextChapter = allChapters.find((c) => c.id === chapter.id + 1);

  return (
    <Layout>
      {/* Header Section */}
      <section className="py-16 bg-gradient-to-br from-primary/10 via-transparent to-primary/5">
        <div className="container">
          <Link href="/book">
            <Button variant="ghost" className="mb-6">
              <ArrowLeft className="mr-2 h-4 w-4" />
              Back to Book
            </Button>
          </Link>
          <div className="max-w-4xl">
            <div className="flex items-center gap-3 mb-4">
              <div className="flex-shrink-0 w-12 h-12 rounded-lg bg-primary/10 flex items-center justify-center">
                <span className="text-xl font-bold text-primary">{chapter.id}</span>
              </div>
              <h1 className="text-4xl md:text-5xl font-bold">{chapter.title}</h1>
            </div>
            <p className="text-xl text-muted-foreground">{chapter.description}</p>
          </div>
        </div>
      </section>

      {/* Content Section */}
      <section className="py-16">
        <div className="container max-w-4xl">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {chapter.sections.map((section, index) => {
              const sectionSlug = section.title.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/(^-|-$)/g, '');
              const sectionKey = `${chapter.slug}-${sectionSlug}`;
              return (
                <Link key={index} href={`/book/chapter/${chapter.slug}/section/${sectionSlug}`}>
                  <Card className="h-full hover:border-primary/50 hover:shadow-lg transition-all cursor-pointer group">
                    <CardHeader>
                      <CardTitle className="text-lg flex items-center justify-between">
                        {section.title}
                        <ChevronRight className="h-5 w-5 text-muted-foreground group-hover:text-primary group-hover:translate-x-1 transition-all" />
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <p className="text-muted-foreground leading-relaxed">{section.content}</p>
                    </CardContent>
                  </Card>
                </Link>
              );
            })}
          </div>

          {/* Download PDF Section */}
          <Card className="mt-12 bg-primary/5 border-primary/20">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <BookOpen className="h-5 w-5" />
                Complete Chapter Content
              </CardTitle>
              <CardDescription>
                For the full detailed content including code examples, diagrams, and in-depth explanations, 
                download the complete book PDF.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <a href="/docs/specification.pdf" download>
                <Button>
                  <Download className="mr-2 h-4 w-4" />
                  Download Complete Book (PDF)
                </Button>
              </a>
            </CardContent>
          </Card>

          {/* Navigation */}
          <div className="flex justify-between items-center mt-12 pt-8 border-t border-border">
            <div>
              {prevChapter && (
                <Link href={`/book/chapter/${prevChapter.slug}`}>
                  <Button variant="outline">
                    <ArrowLeft className="mr-2 h-4 w-4" />
                    Chapter {prevChapter.id}: {prevChapter.title}
                  </Button>
                </Link>
              )}
            </div>
            <div>
              {nextChapter && (
                <Link href={`/book/chapter/${nextChapter.slug}`}>
                  <Button variant="outline">
                    Chapter {nextChapter.id}: {nextChapter.title}
                    <ArrowRight className="ml-2 h-4 w-4" />
                  </Button>
                </Link>
              )}
            </div>
          </div>
        </div>
      </section>
    </Layout>
  );
}
