require 'bibtex'

module Jekyll
  class PublicationsGenerator < Generator
    safe true
    priority :high

    def generate(site)
      # BibTeX 파일 경로
      bib_file = File.join(site.source, '_data', 'publications.bib')
      
      return unless File.exist?(bib_file)

      begin
        # BibTeX 파일 파싱
        bibliography = BibTeX.open(bib_file)
        publications = []

        bibliography.each do |entry|
          pub = {
            'title' => entry.title.to_s,
            'author' => entry.author.to_s,
            'year' => entry.year.to_s,
            'journal' => entry.journal.to_s,
            'booktitle' => entry.booktitle.to_s,
            'volume' => entry.volume.to_s,
            'pages' => entry.pages.to_s,
            'doi' => entry.doi.to_s,
            'url' => entry.url.to_s,
            'type' => entry.type.to_s
          }

          publications << pub
        end

        # site.data에 저장
        site.data['publications'] = publications

      rescue StandardError => e
        Jekyll.logger.warn "Error parsing BibTeX file: #{e.message}"
        site.data['publications'] = []
      end
    end
  end
end
