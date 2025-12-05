import { Link, useLocation } from "wouter";
import { Button } from "@/components/ui/button";
import { Menu, X } from "lucide-react";
import { useState } from "react";

interface LayoutProps {
  children: React.ReactNode;
}

export default function Layout({ children }: LayoutProps) {
  const [location] = useLocation();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const navigation = [
    { name: "Home", href: "/" },
    { name: "Specification", href: "/specification" },
    { name: "Book", href: "/book" },
    { name: "Examples", href: "/examples" },
  ];

  const isActive = (href: string) => {
    if (href === "/") return location === "/";
    return location.startsWith(href);
  };

  return (
    <div className="min-h-screen flex flex-col bg-background text-foreground">
      {/* Navigation */}
      <header className="sticky top-0 z-50 w-full border-b border-border/40 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <nav className="container flex h-16 items-center justify-between">
          <Link href="/" className="flex items-center space-x-3">
            <div className="flex items-center justify-center w-10 h-10 rounded-lg bg-primary">
              <span className="text-primary-foreground font-bold text-xl">TVL</span>
            </div>
            <div className="flex flex-col">
              <span className="font-bold text-lg leading-none">Tuned Variable Language</span>
              <span className="text-xs text-muted-foreground">AI Pipeline Adaptation</span>
            </div>
          </Link>

          {/* Desktop Navigation */}
          <div className="hidden md:flex items-center space-x-1">
            {navigation.map((item) => (
              <Link key={item.name} href={item.href}>
                <Button
                  variant={isActive(item.href) ? "default" : "ghost"}
                  className="text-sm font-medium"
                >
                  {item.name}
                </Button>
              </Link>
            ))}
          </div>

          {/* Mobile menu button */}
          <Button
            variant="ghost"
            size="icon"
            className="md:hidden"
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
          >
            {mobileMenuOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
          </Button>
        </nav>

        {/* Mobile Navigation */}
        {mobileMenuOpen && (
          <div className="md:hidden border-t border-border">
            <div className="container py-4 space-y-2">
              {navigation.map((item) => (
                <Link key={item.name} href={item.href}>
                  <Button
                    variant={isActive(item.href) ? "default" : "ghost"}
                    className="w-full justify-start text-sm font-medium"
                    onClick={() => setMobileMenuOpen(false)}
                  >
                    {item.name}
                  </Button>
                </Link>
              ))}
            </div>
          </div>
        )}
      </header>

      {/* Main Content */}
      <main className="flex-1">{children}</main>

      {/* Footer */}
      <footer className="border-t border-border bg-card">
        <div className="container py-12">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div>
              <h3 className="font-semibold text-lg mb-4">TVL</h3>
              <p className="text-sm text-muted-foreground">
                Specification backbone for governed adaptation in AI pipelines.
              </p>
            </div>
            <div>
              <h3 className="font-semibold text-lg mb-4">Resources</h3>
              <ul className="space-y-2 text-sm">
                <li>
                  <Link href="/specification" className="text-muted-foreground hover:text-foreground transition-colors">
                    Specification
                  </Link>
                </li>
                <li>
                  <Link href="/book" className="text-muted-foreground hover:text-foreground transition-colors">
                    Book
                  </Link>
                </li>
                <li>
                  <Link href="/examples" className="text-muted-foreground hover:text-foreground transition-colors">
                    Examples
                  </Link>
                </li>
              </ul>
            </div>
            <div>
              <h3 className="font-semibold text-lg mb-4">Powered By</h3>
              <a
                href="https://www.traigent.ai/"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-block hover:opacity-80 transition-opacity"
              >
                <img 
                  src="/traigent-logo.webp" 
                  alt="Traigent" 
                  className="h-8 w-auto"
                />
              </a>
              <p className="text-xs text-muted-foreground mt-3">
                AI Agent Optimization Platform
              </p>
            </div>
          </div>
          <div className="mt-8 pt-8 border-t border-border text-center text-sm text-muted-foreground">
            <p>&copy; {new Date().getFullYear()} Traigent. All rights reserved.</p>
          </div>
        </div>
      </footer>
    </div>
  );
}
