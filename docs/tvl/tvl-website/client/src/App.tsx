import { Toaster } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import NotFound from "@/pages/NotFound";
import { Route, Switch } from "wouter";
import ErrorBoundary from "./components/ErrorBoundary";
import { ThemeProvider } from "./contexts/ThemeContext";
import Home from "./pages/Home";
import Specification from "./pages/Specification";
import Book from "./pages/Book";
import Examples from "./pages/Examples";
import Chapter from "./pages/Chapter";
import Section from "./pages/Section";

function Router() {
  return (
    <Switch>
      <Route path={"/"} component={Home} />
         <Route path="/specification" component={Specification} />
      <Route path="/book" component={Book} />
      <Route path="/book/chapter/:chapterSlug" component={Chapter} />
      <Route path="/book/chapter/:chapterSlug/section/:sectionSlug" component={Section} />
      <Route path="/examples" component={Examples} />
      <Route path={"/404"} component={NotFound} />
      {/* Final fallback route */}
      <Route component={NotFound} />
    </Switch>
  );
}

function App() {
  return (
    <ErrorBoundary>
      <ThemeProvider defaultTheme="dark">
        <TooltipProvider>
          <Toaster />
          <Router />
        </TooltipProvider>
      </ThemeProvider>
    </ErrorBoundary>
  );
}

export default App;
