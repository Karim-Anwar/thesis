import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.InputStream;

import org.apache.jena.query.Dataset;
import org.apache.jena.query.DatasetFactory;
import org.apache.jena.rdf.model.Model;
import org.apache.jena.rdfconnection.RDFConnection;
import org.apache.jena.rdfconnection.RDFConnectionFactory;
import org.apache.jena.riot.Lang;
import org.apache.jena.riot.RDFParser;
import org.apache.jena.riot.RDFWriter;

/**
 * Hello world!
 *
 */
public class TypeSummary {
    public static void main(String[] args) throws FileNotFoundException {

        InputStream source = TypeSummary.class.getResourceAsStream(args.length == 0 ? "bgs_stripped.nt" : args[0]);

        if (source == null) {
            throw new Error();
        }

        Dataset dataset = DatasetFactory.create();
        RDFParser.source(source).forceLang(Lang.NT).build().parse(dataset.asDatasetGraph());

        RDFConnection conn = RDFConnectionFactory.connect(dataset);

        Model model = conn.queryConstruct(
                "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> \n"
                + "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> \n"
                + "CONSTRUCT {?subjectType ?predicate ?objectType} \n"
                + "WHERE {"
                + " ?subject a ?subjectType ."
                + "?object a ?objectType ."
                + "?subject ?predicate ?object"
                + "}"
        );

        RDFWriter.create().source(model).lang(Lang.NT).output(new FileOutputStream(new File("newbgs_summarized.nt")));

    }
}
